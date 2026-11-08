"""Frozen state embeddings, text selection and bounded concurrent caching without model downloads."""

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from contextos.config import Config
from contextos.history import NOTE, Group, History, Snapshot
from contextos.observe import WORK_LABELS, Encoder, observation_size, observe


class Tokenizer:
    """A reversible token per character, plus three real special tokens."""

    def encode(self, text, add_special_tokens=False, **kwargs):
        ids = [ord(c) for c in text]
        return [-1, *ids, -2, -3] if add_special_tokens else ids

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids if i >= 0)

    def num_special_tokens_to_add(self, pair=False):
        return 3


class Parameter:
    requires_grad = True

    def requires_grad_(self, value):
        self.requires_grad = value
        return self


@pytest.fixture
def model_class(monkeypatch):
    class Model:
        instances = []
        width = 1024

        def __init__(self, name, **kwargs):
            self.instances.append(self)
            self.name, self.kwargs = name, kwargs
            self.tokenizer = Tokenizer()
            self.weights = [Parameter(), Parameter()]
            self.evaluating = False
            self.max_seq_length = 9999
            self.calls = []
            self.output = None
            self.entered = self.release = None

        def eval(self):
            self.evaluating = True
            return self

        def parameters(self):
            return iter(self.weights)

        def get_sentence_embedding_dimension(self):
            return self.width

        def encode(self, text, **kwargs):
            assert self.evaluating and not any(p.requires_grad for p in self.weights)
            assert kwargs == {
                "prompt": "",
                "normalize_embeddings": False,
                "convert_to_numpy": True,
                "show_progress_bar": False,
            }
            assert len(self.tokenizer.encode(text, add_special_tokens=True)) <= self.max_seq_length
            self.calls.append(text)
            if self.entered is not None:
                self.entered.set()
                assert self.release.wait(5)
            if self.output is not None:
                return self.output
            vector = np.arange(1, self.width + 1, dtype=np.float32)
            vector[0] += sum((i + 1) * ord(c) for i, c in enumerate(text)) % 1009
            return vector

    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = Model
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return Model


def state(*, long=False):
    def text(label):
        return f"{label}_BEGIN " + ((label.lower() + " ") * 5000 if long else "observed facts") + f" {label}_END"

    history = History([{"role": "system", "content": "Pinned instructions"}, {"role": "user", "content": text("TASK")}])
    history.groups.append(Group("summary:installed", [{"role": "user", "content": NOTE + text("MEMORY")}], 0))
    history.append(
        1,
        [
            {
                "role": "assistant",
                "content": text("ACTION"),
                "tool_calls": [
                    {
                        "id": "tool-1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": json.dumps({"command": "pytest test_parser.py"})},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "tool-1", "content": text("EVIDENCE")},
        ],
    )
    return SimpleNamespace(
        history=history,
        t=2,
        horizon=200,
        cfg=SimpleNamespace(input_tokens=32768),
        growth=[1000],
        last_commit=0,
        latest_error=False,
        phase="IDLE",
        job=None,
        size=lambda: 8000,
        prefix_size=lambda groups=None: 4000,
        candidate_size=lambda: 5000,
    )


def sections(block):
    result = {}
    for i, label in enumerate(WORK_LABELS):
        part = block.split(label + "\n", 1)[1]
        if i + 1 < len(WORK_LABELS):
            part = part.split("\n\n" + WORK_LABELS[i + 1] + "\n", 1)[0]
        result[label] = part
    return result


def test_model_is_frozen_revision_pinned_and_configuration_isolated(model_class):
    cfg = Config()
    encoder = Encoder(cfg)
    model = model_class.instances[0]
    assert model.name == cfg.encoder.model
    assert model.kwargs == {"revision": cfg.encoder.revision, "device": "cpu", "trust_remote_code": False}
    assert model.evaluating and not any(p.requires_grad for p in model.weights)
    assert encoder.dimension == 256 and encoder.size == observation_size(cfg) == 524
    assert encoder.content_tokens == 2045 and model.max_seq_length == 2048
    cfg.encoder.dimension, cfg.encoder.max_tokens, cfg.encoder.cache_size = 64, 128, 0
    cfg.encoder.revision = "changed-after-construction"
    assert encoder.dimension == 256 and encoder.cfg.max_tokens == 2048 and encoder.cfg.cache_size == 1024
    assert encoder.cfg.revision == model.kwargs["revision"]


def test_mrl_prefix_is_normalized_and_cache_returns_independent_copies(model_class):
    encoder = Encoder(Config())
    model = model_class.instances[0]
    model.output = np.arange(1, 1025, dtype=np.float32)
    first = encoder.encode("same text")
    expected = np.arange(1, 257, dtype=np.float64)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(first, expected, atol=1e-8)
    assert first.dtype == np.float32 and np.linalg.norm(first) == pytest.approx(1.0)
    first[:] = np.nan
    second = encoder.encode("same text")
    np.testing.assert_allclose(second, expected, atol=1e-8)
    second[:] = 0
    np.testing.assert_allclose(encoder.encode("same text"), expected, atol=1e-8)
    assert model.calls == ["same text"]


def test_cache_is_lru_and_disabled_cache_recomputes(model_class):
    cfg = Config()
    cfg.encoder.cache_size = 2
    encoder = Encoder(cfg)
    for text in ("a", "b", "a", "c", "b"):
        encoder.encode(text)
    assert model_class.instances[0].calls == ["a", "b", "c", "b"]
    assert list(encoder.cache) == ["c", "b"]
    cfg.encoder.cache_size = 0
    uncached = Encoder(cfg)
    uncached.encode("same")
    uncached.encode("same")
    assert model_class.instances[1].calls == ["same", "same"] and not uncached.cache


def test_concurrent_identical_misses_use_one_encoding(model_class):
    encoder = Encoder(Config())
    model = model_class.instances[0]
    model.entered, model.release = threading.Event(), threading.Event()
    barrier = threading.Barrier(9)

    def encode():
        barrier.wait(timeout=5)
        return encoder.encode("shared state")

    pool = ThreadPoolExecutor(max_workers=8)
    try:
        futures = [pool.submit(encode) for _ in range(8)]
        barrier.wait(timeout=5)
        assert model.entered.wait(5)
        model.release.set()
        vectors = [future.result(timeout=5) for future in futures]
    finally:
        model.release.set()
        pool.shutdown(wait=True)
    assert model.calls == ["shared state"]
    for vector in vectors[1:]:
        np.testing.assert_array_equal(vector, vectors[0])
        assert not np.shares_memory(vector, vectors[0])


@pytest.mark.parametrize(
    "invalid",
    [
        np.full(1024, np.nan),
        np.full(1024, np.inf),
        np.ones((1, 1024)),
        np.ones(32),
        np.zeros(1024),
        np.r_[np.zeros(256), np.ones(768)],
        np.r_[np.ones(256), np.full(768, np.nan)],
    ],
)
def test_invalid_output_does_not_poison_the_cache(model_class, invalid):
    encoder = Encoder(Config())
    model = model_class.instances[0]
    model.output = invalid
    with pytest.raises(ValueError):
        encoder.encode("retry this state")
    assert not encoder.cache
    model.output = np.ones(1024)
    vector = encoder.encode("retry this state")
    assert np.isfinite(vector).all() and np.linalg.norm(vector) == pytest.approx(1.0)
    assert len(model.calls) == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("dimension", 0),
        ("dimension", True),
        ("dimension", 1.5),
        ("dimension", 31),
        ("dimension", 2048),
        ("max_tokens", 0),
        ("max_tokens", True),
        ("cache_size", -1),
        ("cache_size", True),
    ],
)
def test_invalid_configuration_fails_before_loading_weights(model_class, field, value):
    cfg = Config()
    setattr(cfg.encoder, field, value)
    with pytest.raises(ValueError, match=r"encoder\."):
        Encoder(cfg)
    assert model_class.instances == []


def test_loaded_width_and_special_token_overhead_are_validated(model_class):
    cfg = Config()
    cfg.encoder.model = "fixture/smaller-model"
    model_class.width = 128
    with pytest.raises(ValueError, match="embedding width"):
        Encoder(cfg)
    model_class.width = 1024
    cfg.encoder.max_tokens = 3
    with pytest.raises(ValueError, match="special tokens"):
        Encoder(cfg)


@pytest.mark.parametrize("max_tokens", [1024, 2048])
def test_work_preserves_all_sections_with_long_inputs_within_total_budget(model_class, max_tokens):
    cfg = Config()
    cfg.encoder.max_tokens = max_tokens
    encoder = Encoder(cfg)
    block = encoder.work(state(long=True))
    selected = sections(block)
    assert all(selected.values())
    assert "TASK_BEGIN" in selected["TASK"] and "TASK_END" in selected["TASK"]
    assert "MEMORY_BEGIN" in selected["INSTALLED MEMORY"] and "MEMORY_END" in selected["INSTALLED MEMORY"]
    assert "ACTION_BEGIN" in selected["LATEST ACTION"]
    assert "EVIDENCE_BEGIN" in selected["RECENT OBSERVATIONS"] and "EVIDENCE_END" in selected["RECENT OBSERVATIONS"]
    tok = encoder.model.tokenizer
    assert len(tok.encode(block, add_special_tokens=True)) <= max_tokens
    limits = [int(limit * min(1.0, max_tokens / 2048)) for limit in (512, 384, 256)]
    for label, limit in zip(WORK_LABELS[:3], limits, strict=True):
        assert len(tok.encode(selected[label])) <= limit
    encoder.encode(block)
    assert encoder.model.calls == [block]


def test_work_uses_installed_memory_and_both_narration_and_actual_tool_call(model_class):
    encoder = Encoder(Config())
    block = encoder.work(state())
    assert "MEMORY_BEGIN observed facts MEMORY_END" in block
    assert "ACTION_BEGIN observed facts ACTION_END" in block
    assert "pytest test_parser.py" in block
    assert "EVIDENCE_BEGIN observed facts EVIDENCE_END" in block
    assert encoder.work(state()) == block


def test_candidates_are_zero_until_ready_and_text_changes_invalidate_cache(model_class):
    encoder = Encoder(Config())
    rt = state()
    first = observe(rt, encoder)
    assert first.shape == (524,) and np.any(first[12:268]) and not np.any(first[268:])
    assert len(encoder.model.calls) == 1
    rt.job = {"snapshot": Snapshot("candidate", 0, [], "", rt.history.groups[-1].id, 2, 4, 0), "summary": None}
    rt.phase = "PREPARING"
    rt.t += 1
    pending = observe(rt, encoder)
    assert not np.any(pending[268:]) and len(encoder.model.calls) == 1
    rt.history.append(2, [{"role": "user", "content": "A boundary"}, {"role": "assistant", "content": "A new action"}])
    forced = observe(rt, encoder)  # PREPARING is a forced CONTINUE state; its work features must still refresh
    assert len(encoder.model.calls) == 2 and not np.array_equal(forced[12:268], pending[12:268])
    rt.phase, rt.job["summary"] = "READY", "candidate memory one"
    ready = observe(rt, encoder)
    assert np.any(ready[268:]) and len(encoder.model.calls) == 3
    np.testing.assert_array_equal(observe(rt, encoder), ready)
    assert len(encoder.model.calls) == 3
    rt.job["summary"] = "candidate memory two"
    changed = observe(rt, encoder)
    assert len(encoder.model.calls) == 4 and not np.array_equal(changed[268:], ready[268:])
    rt.history.groups[0] = Group("summary:next", [{"role": "user", "content": NOTE + "Different installed memory"}], 3)
    changed_memory = observe(rt, encoder)
    assert len(encoder.model.calls) == 5 and not np.array_equal(changed_memory[12:268], changed[12:268])
    assert np.isfinite(changed_memory).all()


def test_candidate_uses_independent_cap_and_cache_keys_exact_selected_text(model_class):
    cfg = Config()
    cfg.encoder.max_tokens = 128
    encoder = Encoder(cfg)
    rt = state()
    rt.phase = "READY"
    rt.job = {
        "snapshot": Snapshot("candidate", 0, [], "", rt.history.groups[-1].id, 2, 4, 0),
        "summary": "CANDIDATE_HEAD" + "x" * 10000 + "CANDIDATE_TAIL",
    }
    result = observe(rt, encoder)
    assert result.shape == (524,)
    assert len(encoder.model.calls) == 2
    work, candidate = encoder.model.calls
    assert candidate.startswith("CANDIDATE_HEAD") and candidate.endswith("CANDIDATE_TAIL")
    assert all(len(encoder.model.tokenizer.encode(text, add_special_tokens=True)) <= 128 for text in (work, candidate))
    rt.job["summary"] = "CANDIDATE_HEAD" + "y" * 10000 + "CANDIDATE_TAIL"
    observe(rt, encoder)
    assert len(encoder.model.calls) == 3  # changed retained head/tail bytes invalidate the candidate
    a = "A" * 100 + "X" * 10000 + "Z" * 100
    b = "A" * 100 + "Y" * 10000 + "Z" * 100
    assert encoder.bounded(a, encoder.content_tokens) == encoder.bounded(b, encoder.content_tokens)
    encoder.encode(a)
    before = len(encoder.model.calls)
    encoder.encode(b)
    assert len(encoder.model.calls) == before  # changes outside the selected text need no new forward pass


def test_bounded_text_rechecks_decode_boundaries_and_small_allowances(model_class):
    encoder = Encoder(Config())

    class ExpandingTokenizer(Tokenizer):
        def decode(self, ids, skip_special_tokens=False):
            return super().decode(ids, skip_special_tokens) * 2

    encoder.model.tokenizer = ExpandingTokenizer()
    for budget in (1, 5, 20, 100):
        selected = encoder.bounded("abcde" * 500, budget)
        assert len(encoder.model.tokenizer.encode(selected)) <= budget
        assert len(encoder.model.tokenizer.encode(selected, add_special_tokens=True)) <= encoder.cfg.max_tokens


def test_numeric_mode_and_empty_text_return_zeros_without_loading(model_class):
    cfg = Config()
    cfg.encoder.semantic = False
    encoder = Encoder(cfg)
    assert model_class.instances == []
    np.testing.assert_array_equal(encoder.encode("ignored"), np.zeros(256))
    assert encoder.work(state()) == ""
    assert not np.any(observe(state(), encoder)[12:])
    semantic = Encoder(Config())
    np.testing.assert_array_equal(semantic.encode(""), np.zeros(256))
    assert semantic.model.calls == []
