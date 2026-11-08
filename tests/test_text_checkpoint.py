"""Feature identity, legacy numeric compatibility, and dense-state learner regressions."""

import copy

import numpy as np
import pytest
import torch

from contextos.config import Config
from contextos.observe import OBSERVATION_SCHEMA_VERSION, SIZE
from contextos.policy import Controller


def config(algorithm="ppo", *, semantic=True, dimension=256):
    cfg = Config()
    cfg.strategy.name = algorithm
    cfg.encoder.semantic, cfg.encoder.dimension = semantic, dimension
    cfg.learner.hidden = [8]
    cfg.learner.choice_only = False
    cfg.learner.async_collection = algorithm == "ppo"
    cfg.learner.epochs = cfg.learner.learning_starts = cfg.learner.transitions_per_update = 1
    cfg.learner.batch_size = 2
    return cfg


def episode(policy, *, reward=1.0, forced_only=False):
    # Nonzero text coordinates verify that the wider observation is carried through learning.
    observations = np.linspace(-0.5, 0.5, policy.observation_size, dtype=np.float32)
    masks = [[True, False, False]] if forced_only else [[True, False, False], [True, True, False]]
    rows = []
    for i, mask in enumerate(masks):
        observation = observations + i * 0.01
        decision = policy.decide(observation, mask)
        rows.append(
            {
                "observation": observation.tolist(),
                "mask": mask,
                "action": int(decision.action),
                "log_prob": decision.log_prob,
                "value": decision.value,
                "version": decision.version,
                "reward": reward if i == len(masks) - 1 else -0.01,
                "terminal": i == len(masks) - 1,
                "next_observation": (observations + (i + 1) * 0.01).tolist(),
                "next_mask": masks[i + 1] if i + 1 < len(masks) else [True, False, False],
            }
        )
    return {"name": "dense-state", "transitions": rows}


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, list | tuple):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_same(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("algorithm", ["ppo", "double_dqn"])
def test_text_checkpoint_restores_dense_state_predictions_and_optimizer(tmp_path, algorithm):
    cfg = config(algorithm)
    original = Controller(cfg)
    metrics = original.update([episode(original)])
    assert metrics["training_transitions"] == 2  # forced states still train the raw-step critic/Q function
    assert metrics["optimizer_steps"] > 0
    original.save(tmp_path / "policy.pt", metadata={"consumed_names": ["dense-state"], "updates": [metrics]})
    saved = torch.load(tmp_path / "policy.pt", weights_only=False)
    assert saved["observation_size"] == 524
    assert saved["observation_identity"] == {
        "scalar_schema": 1,
        "semantic": True,
        "dimension": 256,
        "text_schema": OBSERVATION_SCHEMA_VERSION,
        "model": cfg.encoder.model,
        "revision": cfg.encoder.revision,
        "max_tokens": cfg.encoder.max_tokens,
    }
    # Controller construction does not load the external encoder. These operational settings may
    # change between collection/evaluation hosts without redefining the input features.
    moved = copy.deepcopy(cfg)
    moved.encoder.device, moved.encoder.cache_size = "cuda:0", 0
    moved.experiment.seed += 1
    resumed = Controller(moved)
    resumed.load(tmp_path / "policy.pt", full=True)
    assert_same(saved, resumed.state())
    observation = torch.linspace(-1.0, 1.0, 524)[None]
    with torch.no_grad():
        assert_same(original.net(observation), resumed.net(observation))


@pytest.mark.parametrize("algorithm", ["ppo", "double_dqn"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("semantic", False),
        ("model", "different/embedding-model"),
        ("revision", "different-revision"),
        ("max_tokens", 1024),
    ],
)
def test_same_width_feature_mismatch_fails_before_mutating_any_learner_state(tmp_path, algorithm, field, value):
    cfg = config(algorithm)
    source = Controller(cfg)
    source.update([episode(source)])
    source.save(tmp_path / "policy.pt")
    incompatible = copy.deepcopy(cfg)
    setattr(incompatible.encoder, field, value)
    incompatible.experiment.seed += 1
    receiver = Controller(incompatible)
    receiver.update([episode(receiver, reward=-0.5)])
    receiver.version = 17
    receiver.checkpoint_metadata = {"sentinel": "unchanged"}
    before = copy.deepcopy(receiver.state())
    with pytest.raises(ValueError, match="encoder identity mismatch"):
        receiver.load(tmp_path / "policy.pt", full=True)
    assert_same(before, receiver.state())


@pytest.mark.parametrize("schema", ["scalar_schema", "text_schema"])
def test_feature_preprocessing_version_is_checked_even_when_shape_is_unchanged(tmp_path, schema):
    cfg = config()
    source = Controller(cfg)
    saved = source.state()
    saved["observation_identity"][schema] -= 1
    torch.save(saved, tmp_path / "policy.pt")
    receiver = Controller(cfg)
    before = copy.deepcopy(receiver.state())
    with pytest.raises(ValueError, match="encoder identity mismatch"):
        receiver.load(tmp_path / "policy.pt", full=False)
    assert_same(before, receiver.state())


def test_numeric_checkpoint_ignores_text_only_encoder_options(tmp_path):
    cfg = config(semantic=False)
    source = Controller(cfg)
    source.save(tmp_path / "policy.pt")
    changed = copy.deepcopy(cfg)
    changed.encoder.model, changed.encoder.revision, changed.encoder.max_tokens = "unused/model", None, 128
    changed.encoder.device, changed.encoder.cache_size = "cuda:0", 0
    receiver = Controller(changed)
    receiver.load(tmp_path / "policy.pt", full=True)
    assert receiver.observation_identity == {"scalar_schema": 1, "semantic": False, "dimension": 256}
    assert_same(source.net.state_dict(), receiver.net.state_dict())


@pytest.mark.parametrize("algorithm", ["ppo", "double_dqn"])
@pytest.mark.parametrize("dimension", [256, 1024])
def test_legacy_numeric_checkpoint_requires_explicit_numeric_mode_and_compatible_width(tmp_path, algorithm, dimension):
    cfg = config(algorithm, semantic=False, dimension=dimension)
    source = Controller(cfg)
    source.update([episode(source)])
    saved = source.state()
    saved.pop("observation_identity")
    if dimension == 1024:
        saved.pop("observation_size")  # historical fixed-width checkpoints use the preserved SIZE fallback
    torch.save(saved, tmp_path / "legacy.pt")
    receiver = Controller(cfg)
    receiver.load(tmp_path / "legacy.pt", full=True)
    assert receiver.observation_size == 12 + 2 * dimension
    assert_same(source.net.state_dict(), receiver.net.state_dict())
    assert receiver.steps == source.steps and receiver.version == source.version
    text = copy.deepcopy(cfg)
    text.encoder.semantic = True
    with pytest.raises(ValueError, match="Legacy checkpoint has no encoder identity"):
        Controller(text).load(tmp_path / "legacy.pt", full=False)


@pytest.mark.parametrize("semantic", [False, True])
def test_legacy_default_width_does_not_silently_become_the_new_width(tmp_path, semantic):
    assert SIZE == 2060
    source = Controller(config(semantic=False, dimension=1024))
    saved = source.state()
    saved.pop("observation_identity")
    saved.pop("observation_size")
    torch.save(saved, tmp_path / "legacy.pt")
    receiver = Controller(config(semantic=semantic, dimension=256))
    before = copy.deepcopy(receiver.state())
    with pytest.raises(ValueError, match="observation size mismatch"):
        receiver.load(tmp_path / "legacy.pt", full=True)
    assert_same(before, receiver.state())


def test_unversioned_semantic_checkpoint_cannot_silently_use_new_text_selection(tmp_path):
    cfg = config()
    saved = Controller(cfg).state()
    saved.pop("observation_identity")
    torch.save(saved, tmp_path / "unversioned-text.pt")
    with pytest.raises(ValueError, match="Legacy checkpoint has no encoder identity"):
        Controller(cfg).load(tmp_path / "unversioned-text.pt", full=False)


def test_null_identity_is_not_a_legacy_numeric_checkpoint(tmp_path):
    cfg = config(semantic=False)
    saved = Controller(cfg).state()
    saved["observation_identity"] = None
    torch.save(saved, tmp_path / "invalid.pt")
    with pytest.raises(ValueError, match="encoder identity mismatch"):
        Controller(cfg).load(tmp_path / "invalid.pt", full=True)


@pytest.mark.parametrize("algorithm", ["ppo", "double_dqn"])
def test_forced_raw_state_keeps_dense_text_features_in_value_learning(algorithm):
    cfg = config(algorithm)
    cfg.learner.hidden = []
    policy = Controller(cfg)
    with torch.no_grad():
        for parameter in policy.net.parameters():
            parameter.zero_()
        if policy.target is not None:
            policy.target.load_state_dict(policy.net.state_dict())
    metrics = policy.update([episode(policy, forced_only=True)])
    assert metrics["training_transitions"] == metrics["optimizer_steps"] == 1
    assert metrics["choices"] == 0
    if algorithm == "ppo":
        assert torch.count_nonzero(policy.net.value.weight[0, 12:]) > 0
        assert torch.count_nonzero(policy.net.actions.weight) == 0
    else:
        assert torch.count_nonzero(policy.net.actions.weight[0, 12:]) > 0
        assert torch.count_nonzero(policy.net.actions.weight[1:]) == 0
