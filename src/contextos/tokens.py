"""One counting path for the prompt that is actually sent."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from typing import cast

from contextos.config import Context

API_KEYS = {"role", "content", "tool_calls", "tool_call_id", "name", "reasoning_content"}


def api_messages(messages: list[dict]) -> list[dict]:
    return [{k: v for k, v in m.items() if k in API_KEYS and v is not None} for m in messages]


def template_messages(messages: list[dict]) -> list[dict]:
    """HF chat templates want parsed tool arguments; the wire format keeps JSON strings."""
    clean = deepcopy(api_messages(messages))
    for message in clean:
        for call in message.get("tool_calls") or []:
            args = call["function"].get("arguments")
            if isinstance(args, str):
                call["function"]["arguments"] = json.loads(args)
    return clean


class Counter:
    def __init__(self, cfg: Context, reasoning_effort: str | None = None, template_kwargs: dict | None = None):
        self.cfg, self.tokenizer, self.reasoning_effort = cfg, None, reasoning_effort
        self.template_kwargs = dict(
            template_kwargs or {}
        )  # the same switches the provider sends (enable_thinking, ...)
        if cfg.tokenizer:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)

    def encode(self, text: str) -> list[int]:
        if self.tokenizer is None:
            return list(text.encode("utf-8"))  # byte estimate, not real tokens
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        if self.tokenizer is None:
            return bytes(tokens).decode("utf-8", errors="ignore")
        return cast(str, self.tokenizer.decode(tokens, skip_special_tokens=False))

    def text(self, text: str) -> int:
        return len(self.encode(text))

    def prompt(self, messages: list[dict], tools: list[dict] | None = None, guarded: bool = True) -> int:
        if self.tokenizer is None:
            size = len(
                json.dumps({"messages": api_messages(messages), "tools": tools or []}, ensure_ascii=False).encode(
                    "utf-8"
                )
            )
        else:
            size = self._template_tokens(template_messages(messages), tools)
        return size + (self.cfg.margin_tokens if guarded else 0)

    def _render(self, messages: list[dict], tools: list[dict] | None) -> int:
        extra = {**self.template_kwargs, **({"tools": tools} if tools else {})}
        if self.reasoning_effort is not None:
            extra["reasoning_effort"] = self.reasoning_effort
        # transformers 5 returns a mapping unless return_dict=False; its length would be 2, not a count
        assert self.tokenizer is not None
        encoded = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False, **extra
        )
        return len(cast(Mapping[str, list[int]], encoded)["input_ids"] if hasattr(encoded, "keys") else encoded)

    def _template_tokens(self, messages: list[dict], tools: list[dict] | None) -> int:
        """The history is measured in pieces too: one interaction group, the recent tail, the prefix.
        Some templates (Qwen3.8) refuse a list with no user turn, so such pieces are measured against
        a one-character sentinel user message whose own cost is subtracted."""
        if any(m.get("role") == "user" for m in messages):
            return self._render(messages, tools)
        sentinel = [{"role": "user", "content": "."}]
        return max(0, self._render(sentinel + messages, tools) - self._render(sentinel, tools))

    def truncate(self, text: str, limit: int, marker: str = "\n[TRUNCATED]\n") -> str:
        """Keep head and tail; re-check because decode/encode can shift BPE boundaries."""
        tokens = self.encode(text)
        if len(tokens) <= limit:
            return text
        middle = len(self.encode(marker))
        if limit <= middle:
            return self.decode(tokens[:limit])
        left = (limit - middle) // 2
        right = limit - middle - left
        value = self.decode(tokens[:left]) + marker + self.decode(tokens[-right:])
        while self.text(value) > limit and right > 0:
            right -= 1
            value = self.decode(tokens[:left]) + marker + (self.decode(tokens[-right:]) if right else "")
        return value
