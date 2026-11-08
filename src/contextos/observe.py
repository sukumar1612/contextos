"""Twelve scalars plus two frozen embeddings; their width is configurable."""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from copy import deepcopy

import numpy as np

from contextos.config import Config

DIM = 1024
SIZE = 12 + 2 * DIM
OBSERVATION_SCHEMA_VERSION = 2
WORK_LABELS = ("TASK", "INSTALLED MEMORY", "LATEST ACTION", "RECENT OBSERVATIONS")


def observation_size(cfg: Config) -> int:
    if type(cfg.encoder.dimension) is not int or cfg.encoder.dimension <= 0:
        raise ValueError("encoder.dimension must be a positive integer")
    return 12 + 2 * cfg.encoder.dimension


class Encoder:
    def __init__(self, cfg: Config):
        self.cfg, self.model = deepcopy(cfg.encoder), None
        self.dimension, self.size = self.cfg.dimension, observation_size(cfg)
        if type(self.cfg.max_tokens) is not int or self.cfg.max_tokens <= 0:
            raise ValueError("encoder.max_tokens must be a positive integer")
        if type(self.cfg.cache_size) is not int or self.cfg.cache_size < 0:
            raise ValueError("encoder.cache_size must be a nonnegative integer")
        self.lock = threading.RLock()  # serialize model/tokenizer access and concurrent cache misses
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        if self.cfg.semantic:
            if self.cfg.model == "Qwen/Qwen3-Embedding-0.6B" and not 32 <= self.dimension <= 1024:
                raise ValueError("Qwen3-Embedding-0.6B supports encoder.dimension from 32 through 1024")
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                self.cfg.model, revision=self.cfg.revision, device=self.cfg.device, trust_remote_code=False
            )
            self.model.max_seq_length = self.cfg.max_tokens
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            width = self.model.get_sentence_embedding_dimension()
            if width is not None and self.dimension > width:
                raise ValueError("encoder.dimension exceeds the model's embedding width")
            if self.content_tokens <= 0:
                raise ValueError("encoder.max_tokens leaves no space after tokenizer special tokens")

    @property
    def content_tokens(self) -> int:
        """Text allowance; the model tokenizer supplies its own special-token overhead."""
        with self.lock:
            special = self.model.tokenizer.num_special_tokens_to_add(pair=False) if self.model is not None else 0
            return max(0, self.cfg.max_tokens - special)

    def _count(self, text: str, *, special: bool = False) -> int:
        assert self.model is not None
        return len(self.model.tokenizer.encode(text, add_special_tokens=special, verbose=False))

    def bounded(self, text: str, tokens: int) -> str:
        """Keep head and tail within the encoder's own token budget."""
        with self.lock:
            if self.model is None or not text or tokens <= 0:
                return ""
            tok = self.model.tokenizer
            limit = min(tokens, self.content_tokens)
            ids = tok.encode(text, add_special_tokens=False, verbose=False)
            extra = max(0, self._count(text, special=True) - self.cfg.max_tokens)
            limit = min(limit, len(ids) - extra)
            if limit <= 0:
                return ""
            if len(ids) <= limit:
                return text
            marker = "\n[TRUNCATED]\n"
            overhead = self._count(marker)
            if limit <= overhead:
                left, right, marker = limit, 0, ""
            else:
                left = (limit - overhead) // 2
                right = limit - overhead - left
            while True:
                value = tok.decode(ids[:left], skip_special_tokens=False) + marker
                if right:
                    value += tok.decode(ids[-right:], skip_special_tokens=False)
                excess = max(self._count(value) - limit, self._count(value, special=True) - self.cfg.max_tokens)
                if excess <= 0:
                    return value
                # Decoding and joining can change token boundaries; never rely on a sliced ID count alone.
                if right:
                    right = max(0, right - excess)
                elif left:
                    left = max(0, left - excess)
                else:
                    return ""

    def encode(self, text: str) -> np.ndarray:
        if self.model is None or not text:
            return np.zeros(self.dimension, dtype=np.float32)
        with self.lock:
            text = self.bounded(text, self.content_tokens)
            if not text:
                return np.zeros(self.dimension, dtype=np.float32)
            cached = self.cache.get(text)
            if cached is not None:
                self.cache.move_to_end(text)
                return cached.copy()
            vector = np.asarray(
                self.model.encode(
                    text, prompt="", normalize_embeddings=False, convert_to_numpy=True, show_progress_bar=False
                ),
                dtype=np.float32,
            )
            if vector.ndim != 1 or vector.size < self.dimension:
                raise ValueError("encoder.dimension exceeds the model's embedding width")
            if not np.isfinite(vector).all():
                raise ValueError("Encoder produced a non-finite embedding")
            # Qwen MRL: select the prefix first, then normalize it. Float64 avoids norm overflow.
            vector = vector[: self.dimension].astype(np.float64)
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(norm) or norm == 0:
                raise ValueError("Encoder produced a zero or non-finite embedding norm")
            vector = (vector / norm).astype(np.float32)
            if self.cfg.cache_size:
                self.cache[text] = vector.copy()
                if len(self.cache) > self.cfg.cache_size:
                    self.cache.popitem(last=False)
            return vector

    def work(self, runtime) -> str:
        """Task, installed memory, latest action and recent tool evidence in one bounded state block."""
        if self.model is None:
            return ""
        messages = runtime.history.messages
        assistant: dict = next((m for m in reversed(messages) if m["role"] == "assistant"), {})
        action = assistant.get("content") or ""
        if assistant.get("tool_calls"):
            action += "\n" + json.dumps(assistant["tool_calls"], ensure_ascii=False, sort_keys=True)
        memory = next(
            (
                "\n".join(m.get("content") or "" for m in g.messages)
                for g in reversed(runtime.history.groups)
                if g.id.startswith("summary:")
            ),
            "",
        )
        evidence = "\n".join(m.get("content", "") for m in messages[-12:] if m["role"] == "tool")

        def render(parts):
            return "\n\n".join(f"{label}\n{text}" for label, text in zip(WORK_LABELS, parts, strict=True))

        with self.lock:
            framing = render([""] * len(WORK_LABELS))
            if self._count(framing, special=True) >= self.cfg.max_tokens:
                raise ValueError("encoder.max_tokens is too small for the labeled work state")
            remaining = self.content_tokens - self._count(framing)
            scale = min(1.0, self.cfg.max_tokens / 2048)
            allowances = [int(limit * scale) for limit in (512, 384, 256)]
            parts = []
            for text, allowance in zip((runtime.history.pinned[1]["content"], memory, action), allowances, strict=True):
                part = self.bounded(text, min(allowance, remaining))
                parts.append(part)
                remaining -= self._count(part)
            parts.append(self.bounded(evidence, remaining))
            block = render(parts)
            while (excess := self._count(block, special=True) - self.cfg.max_tokens) > 0:
                # Preserve section labels if BPE boundary merges change the sum of individual counts.
                index = next(i for i in reversed(range(len(parts))) if parts[i])
                parts[index] = self.bounded(parts[index], max(0, self._count(parts[index]) - excess - 1))
                block = render(parts)
            return block


def observe(runtime, encoder: Encoder) -> np.ndarray:
    t, horizon, budget = runtime.t, runtime.horizon, runtime.cfg.input_tokens
    x = np.zeros(12, dtype=np.float32)
    x[0] = runtime.size() / budget  # utilization
    x[1] = np.mean(runtime.growth[-3:]) / budget if runtime.growth else 0  # growth
    x[2] = t / horizon
    x[3] = runtime.prefix_size() / budget  # summarizable prefix
    x[4 + ["IDLE", "PREPARING", "READY"].index(runtime.phase)] = 1
    candidate = ""
    if runtime.job:
        snapshot = runtime.job["snapshot"]
        x[7] = (t - snapshot.started) / horizon
        index = next((i for i, g in enumerate(runtime.history.groups) if g.id == snapshot.tip), None)
        if index is not None:
            x[8] = runtime.prefix_size(runtime.history.groups[index + 1 :]) / budget  # growth since snapshot
        if runtime.phase == "READY":
            candidate = runtime.job["summary"]
            x[9] = (runtime.size() - runtime.candidate_size()) / budget  # gain if committed
    x[10] = (t - runtime.last_commit) / horizon
    x[11] = float(runtime.latest_error)
    x[[0, 1, 3, 8]] = np.clip(x[[0, 1, 3, 8]], 0, 2)
    x[[2, 7, 10]] = np.clip(x[[2, 7, 10]], 0, 1)
    x[9] = np.clip(x[9], -2, 2)
    result = np.concatenate(
        [
            x,
            encoder.encode(encoder.work(runtime)),
            encoder.encode(candidate),
        ]
    )
    if result.shape != (encoder.size,) or not np.isfinite(result).all():
        raise ValueError("Invalid observation")
    return result
