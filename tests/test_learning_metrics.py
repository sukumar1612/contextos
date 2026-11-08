"""Decision-level credit, compact states, and grouped-return update regressions."""

import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from contextos.config import Config
from contextos.observe import Encoder, observation_size
from contextos.policy import Controller, Rule, choice_transitions, gae
from contextos.types import Action


def config(algorithm="ppo"):
    cfg = Config()
    cfg.strategy.name = algorithm
    cfg.encoder.semantic, cfg.encoder.dimension = False, 256
    cfg.learner.hidden = [16]
    cfg.learner.choice_only = True
    cfg.learner.epochs = 1
    return cfg


def trajectory(controller, rewards, masks):
    rows = []
    for reward, mask in zip(rewards, masks, strict=False):
        observation = np.zeros(controller.observation_size, np.float32)
        decision = controller.decide(observation, mask)
        rows.append(
            {
                "observation": observation.tolist(),
                "mask": mask,
                "action": int(decision.action),
                "log_prob": decision.log_prob,
                "value": decision.value,
                "version": decision.version,
                "reward": reward,
                "terminal": False,
                "next_observation": observation.tolist(),
            }
        )
    for i, row in enumerate(rows):
        row["terminal"] = i == len(rows) - 1
        row["next_mask"] = masks[i + 1] if not row["terminal"] else [True, False, False]
    return rows


def test_compact_encoder_normalizes_after_truncating_and_checkpoint_checks_shape(tmp_path):
    cfg = config()
    encoder = Encoder(cfg)
    assert encoder.encode("").shape == (256,)
    assert observation_size(cfg) == encoder.size == 524

    class Tokenizer:
        def encode(self, text, add_special_tokens=False, **kwargs):
            return [ord(c) for c in text]

        def decode(self, ids, skip_special_tokens=False):
            return "".join(chr(i) for i in ids)

        def num_special_tokens_to_add(self, pair=False):
            return 0

    class Model:
        tokenizer = Tokenizer()

        def encode(self, *args, **kwargs):
            return np.arange(1, 1025, dtype=np.float32)

    encoder.model = Model()
    vector = encoder.encode("text")
    assert vector.shape == (256,)
    assert np.linalg.norm(vector) == pytest.approx(1.0)
    np.testing.assert_allclose(vector, np.arange(1, 257) / np.linalg.norm(np.arange(1, 257)), atol=1e-8)
    controller = Controller(cfg)
    assert controller.net.body[0].in_features == 524
    controller.save(tmp_path / "compact.pt")
    cfg.encoder.dimension = 1024
    with pytest.raises(ValueError, match="observation size"):
        Controller(cfg).load(tmp_path / "compact.pt", full=False)


def test_checkpoint_keeps_experience_order_atomic_with_weights(tmp_path):
    cfg = config("double_dqn")
    controller = Controller(cfg)
    controller.version, controller.inserted = 3, 7
    metadata = {"learned_names": ["episode-b", "episode-a"], "updates": [{"version": 3}]}
    controller.save(tmp_path / "policy.pt", metadata=metadata)
    metadata["learned_names"].clear()
    resumed = Controller(cfg)
    resumed.load(tmp_path / "policy.pt", full=True)
    assert (resumed.version, resumed.inserted) == (3, 7)
    assert resumed.checkpoint_metadata["learned_names"] == ["episode-b", "episode-a"]
    assert resumed.checkpoint_metadata["updates"] == [{"version": 3}]


@pytest.mark.parametrize("gamma", [1.0, 0.5])
def test_collapsing_forced_steps_preserves_delayed_outcome_and_discount(gamma):
    controller = Controller(config())
    masks = [[True, False, False], [True, True, False], [True, False, False], [True, False, True], [True, False, False]]
    rows = trajectory(controller, [-0.1, -0.2, -0.3, -0.4, 1.0], masks)
    original = copy.deepcopy(rows)
    settled = choice_transitions(rows, gamma)
    assert rows == original  # raw episode accounting is untouched
    assert len(settled) == 2
    assert settled[0]["reward"] == pytest.approx(-0.2 - gamma * 0.3)
    assert settled[1]["reward"] == pytest.approx(-0.4 + gamma)
    assert settled[0]["duration"] == settled[1]["duration"] == 2
    assert settled[0]["next_mask"] == [True, False, True]
    assert settled[-1]["terminal"]
    # With lambda=1, both time representations have the same return at a choice.
    _, raw_targets = gae(rows, gamma, 1.0)
    _, choice_targets = gae(settled, gamma, 1.0)
    np.testing.assert_allclose(choice_targets, raw_targets[[1, 3]], atol=1e-6)


@pytest.mark.parametrize("algorithm", ["ppo", "double_dqn"])
def test_forced_only_episode_does_not_inflate_learning_data(algorithm):
    controller = Controller(config(algorithm))
    rows = trajectory(controller, [-0.1, 1.0], [[True, False, False]] * 2)
    before = copy.deepcopy(controller.net.state_dict())
    result = controller.update([{"transitions": rows}])
    assert result["choices"] == result["training_transitions"] == result["optimizer_steps"] == 0
    assert result["raw_transitions"] == 2
    assert all(torch.equal(value, before[key]) for key, value in controller.net.state_dict().items())


def test_ddqn_bootstrap_masks_illegal_best_action_and_uses_elapsed_discount():
    cfg = config("double_dqn")
    cfg.learner.choice_only = False
    cfg.learner.hidden = []
    cfg.learner.gamma = 0.5
    cfg.learner.batch_size = cfg.learner.learning_starts = cfg.learner.transitions_per_update = 1
    controller = Controller(cfg)
    with torch.no_grad():
        controller.net.actions.weight.zero_()
        controller.target.actions.weight.zero_()
        controller.net.actions.bias.copy_(torch.tensor([1.0, 100.0, 2.0]))
        controller.target.actions.bias.copy_(torch.tensor([5.0, 90.0, 7.0]))
    row = trajectory(controller, [1.0], [[True, True, False]])[0]
    row.update(action=0, terminal=False, next_mask=[True, False, True], duration=3)
    result = controller.update([{"transitions": [row]}])
    # Online chooses legal action 2; target evaluates it as 7. target=1 + 0.5^3*7.
    assert result["loss"] == pytest.approx(0.5 * (1.875 - 1.0) ** 2)
    assert result["optimizer_steps"] == 1


def test_ddqn_warmup_and_target_clock_count_settled_choices():
    cfg = config("double_dqn")
    cfg.learner.learning_starts = cfg.learner.target_interval = 2
    cfg.learner.transitions_per_update = 1
    cfg.learner.target_clock = "choices"
    controller = Controller(cfg)
    masks = [[True, True, False]] + [[True, False, False]] * 3
    first = controller.update([{"transitions": trajectory(controller, [-0.1] * 3 + [1.0], masks)}])
    assert first["training_transitions"] == first["replay_size"] == 1
    assert first["optimizer_steps"] == 0
    second = controller.update([{"transitions": trajectory(controller, [-0.1] * 3 + [1.0], masks)}])
    assert second["optimizer_steps"] == 1
    assert second["replay_size"] == 2
    assert all(
        torch.equal(value, controller.target.state_dict()[key]) for key, value in controller.net.state_dict().items()
    )


def test_sync_baseline_never_prepares_opportunistically():
    cfg = Config()
    cfg.strategy.name = "sync_at_limit"
    cfg.strategy.prepare_fraction = 0.1
    rule = Rule(cfg)
    for mask in ([True, True, False], [True, False, True]):
        assert rule.decide([1.2], mask).action == Action.CONTINUE


def collector_config(tmp_path, algorithm):
    from contextos.cli import demo_config

    cfg = demo_config(tmp_path, algorithm)
    cfg.encoder.dimension = 256
    cfg.learner.choice_only = True
    cfg.experiment.workers = 2
    rows = [
        {
            "instance_id": f"fixture__repo-{i}",
            "repo": "fixture/repo",
            "base_commit": "a" * 40,
            "problem_statement": f"Fix deterministic fixture issue {i}.",
        }
        for i in range(4)
    ]
    Path(cfg.benchmark.rows).write_text("".join(json.dumps(row) + "\n" for row in rows))
    return cfg


def test_ppo_choice_target_pause_keeps_one_frozen_policy_until_settlement(tmp_path):
    from contextos.experiments import run

    cfg = collector_config(tmp_path, "ppo")
    cfg.learner.choices_per_update = 10_000  # more than the entire small fixture can supply
    cfg.learner.episodes_per_update = 2
    directory = tmp_path / "run"
    first = run(cfg, directory, True, "train", max_episodes=2)
    assert first["updates"] == 0
    saved = torch.load(directory / "policy.pt", weights_only=False)
    assert saved["version"] == saved["steps"] == 0  # paused actors' RNG is checkpointed before the first update
    second = run(cfg, directory, True, "train")
    assert second["updates"] == 1 and second["finished"]
    for path in (directory / "episodes").glob("*.json.gz"):
        with gzip.open(path, "rt") as stream:
            episode = json.load(stream)
        assert {row["version"] for row in episode["transitions"]} == {0}
        assert all(len(row["observation"]) == len(row["next_observation"]) == 524 for row in episode["transitions"])


def test_ddqn_resume_uses_checkpoint_metadata_when_progress_lags(tmp_path):
    from contextos.experiments import run

    cfg = collector_config(tmp_path, "double_dqn")
    directory = tmp_path / "run"
    run(cfg, directory, True, "train")
    saved = torch.load(directory / "policy.pt", weights_only=False)
    progress = json.loads((directory / "progress.json").read_text())
    consumed = list(saved["metadata"]["learned_names"])
    progress["learned_names"] = consumed[:-1]
    progress["updates"] = progress["updates"][:-1]
    progress["pending"] = consumed[-1:]
    (directory / "progress.json").write_text(json.dumps(progress))
    result = run(cfg, directory, True, "train")
    assert result["finished"] and result["ran_now"] == 0 and result["updates"] == 4
    restored = json.loads((directory / "progress.json").read_text())
    assert restored["learned_names"] == consumed and restored["pending"] == []
    after = torch.load(directory / "policy.pt", weights_only=False)
    assert after["version"] == saved["version"] and after["inserted"] == saved["inserted"]
    assert all(torch.equal(value, after["network"][key]) for key, value in saved["network"].items())


def test_ddqn_resume_learns_final_pending_episode_even_when_schedule_ended(tmp_path, monkeypatch):
    from contextos.experiments import Collector, run

    cfg = collector_config(tmp_path, "double_dqn")
    directory = tmp_path / "run"
    original, calls = Collector.learn, []

    def crash_before_last_update(self):
        calls.append(1)
        if len(calls) == 4:
            raise RuntimeError("simulated stop before the final learner update")
        return original(self)

    monkeypatch.setattr(Collector, "learn", crash_before_last_update)
    with pytest.raises(RuntimeError, match="simulated stop"):
        run(cfg, directory, True, "train")
    progress = json.loads((directory / "progress.json").read_text())
    assert progress["cursor"] == 4 and len(progress["pending"]) == 1 and not progress["in_flight"]
    monkeypatch.setattr(Collector, "learn", original)
    resumed = run(cfg, directory, True, "train")
    assert resumed["finished"] and resumed["ran_now"] == 0 and resumed["updates"] == 4
    assert json.loads((directory / "progress.json").read_text())["pending"] == []


def test_async_ppo_learns_from_cohorts_while_episodes_keep_running(tmp_path):
    """Asynchronous PPO: workers never wait for a cohort; later episodes span policy versions, and the
    decoupled update accepts them instead of refusing a mixed-version batch."""
    import gzip
    import json

    from contextos.cli import demo_config
    from contextos.experiments import run

    cfg = demo_config(tmp_path, "ppo")
    cfg.experiment.workers, cfg.learner.episodes_per_update, cfg.learner.async_collection = 4, 4, True
    cfg.experiment.train_tasks = 2
    Path(cfg.benchmark.rows).write_text(
        "".join(
            json.dumps(
                {
                    "instance_id": f"fixture__repo-{i}",
                    "repo": "fixture/repo",
                    "base_commit": "a" * 40,
                    "problem_statement": "Fix it.",
                }
            )
            + "\n"
            for i in range(4)
        )
    )
    result = run(cfg, tmp_path / "run", True, "train")  # 2 train tasks x 4 attempts = 8 episodes, 2 cohorts
    progress = json.loads((tmp_path / "run" / "progress.json").read_text())
    assert result["episodes"] == 8 and len(progress["updates"]) == 2 and progress["pending"] == []
    assert all("max_staleness" in u and "dropped_stale_episodes" in u for u in progress["updates"])
    versions = {
        t["version"]
        for f in (tmp_path / "run" / "episodes").glob("*.json.gz")
        for t in json.load(gzip.open(f, "rt"))["transitions"]
    }
    assert versions >= {0, 1}  # the second cohort's episodes decided under both versions


def test_decoupled_update_drops_episodes_beyond_the_staleness_threshold(tmp_path):
    import numpy as np

    from contextos.cli import demo_config
    from contextos.policy import Controller, observation_size

    cfg = demo_config(tmp_path, "ppo")
    cfg.learner.staleness_threshold, cfg.learner.async_collection = 1, True
    controller = Controller(cfg)
    size = observation_size(cfg)

    def episode(version):
        transitions = []
        for i in range(3):
            decision = controller.decide(np.full(size, 0.1 * i, np.float32), [True, True, False], training=True)
            transitions.append(
                {
                    "observation": [0.1 * i] * size,
                    "next_observation": [0.1 * (i + 1)] * size,
                    "mask": [True, True, False],
                    "next_mask": [True, True, False],
                    "action": int(decision.action),
                    "log_prob": decision.log_prob,
                    "value": decision.value,
                    "version": version,
                    "reward": 0.5 if i == 2 else -0.1,
                    "terminal": i == 2,
                }
            )
        return {"transitions": transitions}

    fresh, stale = episode(0), episode(0)
    controller.version = 1
    metrics = controller.update([fresh])  # one update behind: kept, corrected by the importance weight
    assert (
        metrics["dropped_stale_episodes"] == 0
        and metrics["max_staleness"] == 1
        and metrics["training_transitions"] == 3
    )
    metrics = controller.update([stale])  # now two behind the current version: dropped
    assert metrics["dropped_stale_episodes"] == 1 and metrics["training_transitions"] == 0
