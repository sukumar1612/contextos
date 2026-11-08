"""The standalone real-encoder check reports failures without requiring model downloads here."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from contextos.config import Config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def smoke(monkeypatch):
    spec = importlib.util.spec_from_file_location("encoder_smoke", ROOT / "audit/encoder_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            ids = list(text.encode("utf-8"))
            return [1, *ids, 2] if add_special_tokens else ids

        def num_special_tokens_to_add(self, pair=False):
            return 2

    class FakeEncoder:
        instances = []
        broken = None

        def __init__(self, cfg):
            self.cfg = cfg.encoder
            self.calls, self.cache = 0, {}
            self.model = SimpleNamespace(
                training=self.broken == "training",
                parameters=lambda: iter(
                    [SimpleNamespace(requires_grad=self.broken == "trainable", dtype=module.torch.bfloat16)]
                ),
                tokenizer=Tokenizer(),
                device=self.cfg.device,
            )
            self.instances.append(self)

        @property
        def content_tokens(self):
            return self.cfg.max_tokens - self.model.tokenizer.num_special_tokens_to_add(pair=False)

        def bounded(self, text, limit):
            return text[:limit]

        def work(self, runtime):
            assert runtime.history.groups[0].id.startswith("summary:")
            assert any(message["role"] == "tool" for message in runtime.history.messages)
            return "TASK\nCheck parser behavior.\nMEMORY\nBoundary case reproduced.\nEVIDENCE\n12 passed."

        def encode(self, text):
            if text not in self.cache:
                self.calls += 1
                vector = np.arange(1, self.cfg.dimension + 1, dtype=np.float32)
                vector /= np.linalg.norm(vector)
                if self.broken == "nonfinite":
                    vector[0] = np.nan
                elif self.broken == "unnormalized":
                    vector *= 2
                elif self.broken == "wrong_shape":
                    vector = vector[:-1]
                self.cache[text] = vector
            result = self.cache[text].copy()
            if self.broken == "unstable":
                self.cache[text] = -result
            return result

    monkeypatch.setattr(module, "Encoder", FakeEncoder)
    monkeypatch.setattr(
        module,
        "version",
        lambda name: {"sentence-transformers": "fixture-st", "transformers": "fixture-transformers"}[name],
    )
    return module, FakeEncoder


def config():
    cfg = Config()
    cfg.encoder.semantic, cfg.encoder.dimension, cfg.encoder.max_tokens = True, 256, 128
    return cfg


def test_encoder_check_reports_real_shapes_caps_and_cache_reuse(smoke):
    module, encoder = smoke
    result = module.check(config())
    assert result["passed"] and result["frozen_eval"] and result["cache_enabled"]
    assert result["embedding_dimension"] == 256 and result["observation_size"] == 524
    assert result["model_dtype"] == "torch.bfloat16"
    assert result["torch_version"] == str(module.torch.__version__)
    assert result["torch_num_threads"] == module.torch.get_num_threads()
    assert result["sentence_transformers_version"] == "fixture-st"
    assert result["transformers_version"] == "fixture-transformers"
    assert encoder.instances[0].calls == 3  # each repeated encoding uses its cached vector
    assert {row["text"] for row in result["measurements"]} == {"work", "candidate", "near_token_cap"}
    for row in result["measurements"]:
        assert row["dimension"] == 256 and row["repeat_exact_match"]
        assert 0 < row["input_tokens_including_special"] <= 128
        assert row["uncached_encode_seconds"] >= 0 and row["repeat_encode_seconds"] >= 0
    assert result["measurements"][-1]["input_tokens_including_special"] >= 120


@pytest.mark.parametrize("broken", ["training", "trainable", "nonfinite", "unnormalized", "wrong_shape", "unstable"])
def test_encoder_check_rejects_invalid_inference(smoke, broken):
    module, encoder = smoke
    encoder.broken = broken
    with pytest.raises(ValueError):
        module.check(config())


def test_encoder_check_cannot_pass_without_text_features(smoke):
    module, encoder = smoke
    cfg = config()
    cfg.encoder.semantic = False
    with pytest.raises(ValueError, match="semantic=true"):
        module.check(cfg)
    assert not encoder.instances


def test_encoder_check_rejects_accidentally_short_long_input(smoke, monkeypatch):
    module, encoder = smoke
    monkeypatch.setattr(encoder, "bounded", lambda self, text, limit: text[:16])
    with pytest.raises(ValueError, match="at least 90%"):
        module.check(config())


def test_encoder_check_cli_applies_config_overlays_and_writes_json(smoke, tmp_path, capsys):
    module, encoder = smoke
    first, second = tmp_path / "first.yaml", tmp_path / "second.yaml"
    first.write_text("encoder:\n  semantic: true\n  dimension: 128\n  max_tokens: 128\n")
    second.write_text("encoder:\n  dimension: 512\n")
    output = tmp_path / "nested" / "check.json"
    module.main(
        ["--config", str(first), "--config", str(second), "--set", "encoder.dimension=256", "--output", str(output)]
    )
    result = json.loads(output.read_text())
    assert json.loads(capsys.readouterr().out) == result
    assert result["embedding_dimension"] == 256 and result["max_tokens"] == 128
    assert len(encoder.instances) == 1
