"""Text-feature identity is immutable across resume; encoder placement and cache size are movable."""

import copy
import json

import pytest

from contextos.config import Config
from contextos.experiments import fingerprint


@pytest.fixture
def run_identity(tmp_path):
    row = {"instance_id": "fixture__repo-1", "problem_statement": "Preserve the parser's final identifier."}
    dataset = tmp_path / "rows.jsonl"
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
    cfg = Config()
    cfg.encoder.semantic, cfg.encoder.dimension, cfg.encoder.max_tokens = True, 256, 2048
    cfg.encoder.device, cfg.encoder.cache_size = "cpu", 1024
    cfg.benchmark.rows = str(dataset)
    return cfg, [(row, 0), (row, 1)]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("semantic", False),
        ("dimension", 128),
        ("model", "fixture/different-embedding-model"),
        ("revision", "f" * 40),
        ("max_tokens", 1024),
    ],
)
def test_text_feature_changes_require_a_new_run_identity(run_identity, field, value):
    cfg, schedule = run_identity
    original = fingerprint(cfg, schedule)
    changed = copy.deepcopy(cfg)
    setattr(changed.encoder, field, value)
    assert fingerprint(changed, schedule) != original
    assert fingerprint(cfg, schedule) == original


@pytest.mark.parametrize(("field", "value"), [("device", "cuda:0"), ("cache_size", 0), ("cache_size", 32)])
def test_encoder_placement_and_cache_changes_preserve_run_identity(run_identity, field, value):
    cfg, schedule = run_identity
    original = fingerprint(cfg, schedule)
    changed = copy.deepcopy(cfg)
    setattr(changed.encoder, field, value)
    assert fingerprint(changed, schedule) == original


def test_text_and_numeric_ablations_with_equal_width_have_distinct_identities(run_identity):
    semantic, schedule = run_identity
    numeric = copy.deepcopy(semantic)
    numeric.encoder.semantic = False
    assert semantic.encoder.dimension == numeric.encoder.dimension == 256
    assert fingerprint(semantic, schedule) != fingerprint(numeric, schedule)
