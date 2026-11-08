"""Async update cadence and crash recovery must preserve already published experience."""

import gzip
import json
import weakref
from pathlib import Path

import numpy as np
import pytest
import torch

from contextos import collector, experiments
from contextos.cli import demo_config


def config(tmp_path, algorithm="ppo"):
    cfg = demo_config(tmp_path, algorithm)
    cfg.encoder.dimension = 256
    cfg.learner.hidden = [8]
    cfg.learner.choice_only = True
    cfg.learner.async_collection = algorithm == "ppo"
    cfg.experiment.workers = 1
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
    return cfg


def install_episodes(monkeypatch, *, stale_after_first=False):
    """One genuine choice plus a forced terminal step; no server, sandbox or background jobs."""
    calls = []

    def attempt(cfg, row, name, directory, provider, counter, encoder, strategy, training):
        calls.append(name)
        transitions = []
        for i, mask in enumerate(([True, True, False], [True, False, False])):
            observation = np.full(strategy.observation_size, 0.1 * i, np.float32)
            decision = strategy.decide(observation, mask, training)
            transitions.append(
                {
                    "observation": observation.tolist(),
                    "next_observation": observation.tolist(),
                    "mask": mask,
                    "next_mask": [True, False, False],
                    "action": int(decision.action),
                    "log_prob": decision.log_prob,
                    "value": decision.value,
                    "version": 0 if stale_after_first else decision.version,
                    "reward": 1.0 if i else -0.1,
                    "terminal": bool(i),
                }
            )
        return {
            "transitions": transitions,
            "events": [],
            "return": 0.9,
            "resolved": True,
            "cost_usd": 0.0,
            "termination": "SUBMITTED",
        }, None

    monkeypatch.setattr(collector, "attempt", attempt)
    return calls


def progress(directory):
    return json.loads((directory / "progress.json").read_text())


def checkpoint(directory):
    return torch.load(directory / "policy.pt", weights_only=False)


@pytest.mark.parametrize(
    "episodes_per_update,choices_per_update,cohorts",
    [(1, 3, [3, 1]), (10, 1, [1, 1, 1, 1]), (2, 0, [2, 2])],
)
def test_async_choice_target_replaces_episode_trigger(
    tmp_path, monkeypatch, episodes_per_update, choices_per_update, cohorts
):
    cfg = config(tmp_path)
    cfg.learner.episodes_per_update = episodes_per_update
    cfg.learner.choices_per_update = choices_per_update
    install_episodes(monkeypatch)
    directory = tmp_path / "run"
    result = experiments.run(cfg, directory, True, "train")
    updates = progress(directory)["updates"]
    assert result["finished"] and result["genuine_choices"] == 4
    assert [u["received_episodes"] for u in updates] == cohorts
    assert [u["choices"] for u in updates] == cohorts  # forced steps do not satisfy the choice trigger
    assert all(u["learner_seconds"] >= 0 for u in updates)


def test_consumption_ledger_includes_stale_drops_without_counting_them_as_trained(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg.learner.episodes_per_update = 1
    cfg.learner.staleness_threshold = 0
    cfg.experiment.attempts_per_task = 2
    calls = install_episodes(monkeypatch, stale_after_first=True)
    directory = tmp_path / "run"
    experiments.run(cfg, directory, True, "train")
    state = progress(directory)
    assert state["consumed_names"] == state["learned_names"] == calls
    assert [u["used_episodes"] for u in state["updates"]] == [1, 0]
    assert [u["dropped_stale_episodes"] for u in state["updates"]] == [0, 1]
    assert state["updates"][1]["dropped_stale_episode_names"] == calls[1:]
    assert state["updates"][1]["used_episode_names"] == []
    assert [u["version"] for u in state["updates"]] == [1, 1]
    resumed = experiments.run(cfg, directory, True, "train", max_episodes=0)
    assert resumed["ran_now"] == 0 and resumed["updates"] == 2
    assert checkpoint(directory)["metadata"]["consumed_names"] == calls


@pytest.mark.parametrize("algorithm", ["ppo", "double_dqn"])
def test_resume_recovers_published_episode_before_progress_commit(tmp_path, monkeypatch, algorithm):
    cfg = config(tmp_path, algorithm)
    calls = install_episodes(monkeypatch)
    directory = tmp_path / "run"
    save_progress = collector.Collector.save_progress

    def fail_after_publication(self):
        if self.pending:
            raise OSError("progress publication failed")
        return save_progress(self)

    monkeypatch.setattr(collector.Collector, "save_progress", fail_after_publication)
    with pytest.raises(OSError, match="progress publication failed"):
        experiments.run(cfg, directory, True, "train", max_episodes=1)
    old = progress(directory)
    assert old["cursor"] == 1 and old["in_flight"] == [0] and old["pending"] == []
    episode_path = next((directory / "episodes").glob("*.json.gz"))
    published = episode_path.read_bytes()
    monkeypatch.setattr(collector.Collector, "save_progress", save_progress)
    resumed = experiments.run(cfg, directory, True, "train", max_episodes=0)
    assert resumed["ran_now"] == 0 and resumed["updates"] == 1
    assert len(calls) == 1 and episode_path.read_bytes() == published
    assert progress(directory)["in_flight"] == progress(directory)["pending"] == []
    assert checkpoint(directory)["metadata"]["consumed_names"] == calls


@pytest.mark.parametrize("algorithm", ["ppo", "double_dqn"])
def test_checkpoint_consumption_wins_over_stale_cursor_and_inflight(tmp_path, monkeypatch, algorithm):
    cfg = config(tmp_path, algorithm)
    calls = install_episodes(monkeypatch)
    directory = tmp_path / "run"
    experiments.run(cfg, directory, True, "train")
    saved = checkpoint(directory)
    before = {p.name: p.read_bytes() for p in (directory / "episodes").glob("*.json.gz")}
    state = progress(directory)
    state.update(cursor=0, in_flight=[0], pending=calls[:1], consumed_names=[], learned_names=[], updates=[])
    (directory / "progress.json").write_text(json.dumps(state))
    resumed = experiments.run(cfg, directory, True, "train", max_episodes=0)
    assert resumed["finished"] and resumed["ran_now"] == 0
    assert len(calls) == 4
    assert progress(directory)["consumed_names"] == saved["metadata"]["consumed_names"]
    assert {p.name: p.read_bytes() for p in (directory / "episodes").glob("*.json.gz")} == before
    after = checkpoint(directory)
    assert after["version"] == saved["version"]
    assert all(torch.equal(value, after["network"][key]) for key, value in saved["network"].items())


def test_resume_requires_weights_for_committed_updates(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    install_episodes(monkeypatch)
    directory = tmp_path / "run"
    experiments.run(cfg, directory, True, "train")
    (directory / "policy.pt").unlink()
    with pytest.raises(FileNotFoundError, match="committed learner updates"):
        experiments.run(cfg, directory, True, "train")


def test_ddqn_resume_requires_authoritative_replay_files(tmp_path, monkeypatch):
    cfg = config(tmp_path, "double_dqn")
    install_episodes(monkeypatch)
    directory = tmp_path / "run"
    experiments.run(cfg, directory, True, "train")
    latest = checkpoint(directory)["metadata"]["consumed_names"][-1]
    (directory / "episodes" / f"{latest}.json.gz").unlink()
    with pytest.raises(FileNotFoundError, match="Cannot rebuild replay: consumed episode"):
        experiments.run(cfg, directory, True, "train")


def test_resume_requeues_missing_unconsumed_episode(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg.learner.async_collection = False
    calls = install_episodes(monkeypatch)
    directory = tmp_path / "run"
    experiments.run(cfg, directory, True, "train", max_episodes=1)
    pending = progress(directory)["pending"]
    assert pending == calls and len(calls) == 1
    episode_path = directory / "episodes" / f"{pending[0]}.json.gz"
    episode_path.unlink()
    resumed = experiments.run(cfg, directory, True, "train", max_episodes=1)
    assert resumed["ran_now"] == 1 and resumed["cursor"] == 1 and episode_path.exists()
    assert len(calls) == 2 and calls[0] == calls[1]
    assert progress(directory)["pending"] == pending


def test_recovered_hole_before_schedule_cap_is_not_blocked_by_later_inflight(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg.learner.async_collection = False
    cfg.learner.choices_per_update = 1000
    cfg.experiment.attempts_per_task = 8
    calls = install_episodes(monkeypatch)
    directory = tmp_path / "run"
    experiments.run(cfg, directory, True, "train", max_episodes=6)
    state = progress(directory)
    original = calls.copy()
    for index in (2, 5):
        (directory / "episodes" / f"{original[index]}.json.gz").unlink()
    state["in_flight"] = [5]
    (directory / "progress.json").write_text(json.dumps(state))
    resumed = experiments.run(cfg, directory, True, "train", schedule_cap=4)
    assert resumed["ran_now"] == 1 and resumed["cursor"] == 6
    assert calls == [*original, original[2]]
    assert progress(directory)["in_flight"] == [5]
    assert not (directory / "episodes" / f"{original[5]}.json.gz").exists()


def test_clean_pause_before_first_ppo_update_preserves_rng_and_choices(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg.learner.async_collection = False
    cfg.learner.choices_per_update = 1000
    cfg.learner.episodes_per_update = 2
    install_episodes(monkeypatch)
    straight, resumed = tmp_path / "straight", tmp_path / "resumed"
    experiments.run(cfg, straight, True, "train")
    first = experiments.run(cfg, resumed, True, "train", max_episodes=2)
    paused = checkpoint(resumed)
    assert first["updates"] == paused["version"] == paused["steps"] == 0
    assert paused["choice_points"] == 2 and paused["metadata"]["consumed_names"] == []
    experiments.run(cfg, resumed, True, "train")
    a, b = checkpoint(straight), checkpoint(resumed)
    assert a["choice_points"] == b["choice_points"] == 4
    assert torch.equal(a["torch_rng"], b["torch_rng"])
    assert all(torch.equal(value, b["network"][key]) for key, value in a["network"].items())
    for a_path in (straight / "episodes").glob("*.json.gz"):
        with gzip.open(a_path, "rt") as a_stream, gzip.open(resumed / "episodes" / a_path.name, "rt") as b_stream:
            assert json.load(a_stream)["transitions"] == json.load(b_stream)["transitions"]


def test_existing_artifacts_require_progress_identity(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    install_episodes(monkeypatch)
    directory = tmp_path / "run"
    experiments.run(cfg, directory, True, "train")
    (directory / "progress.json").unlink()
    with pytest.raises(ValueError, match="no progress.json to validate"):
        experiments.run(cfg, directory, True, "train")


def test_report_does_not_retain_the_entire_campaign(tmp_path, monkeypatch):
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    for i in range(6):
        with gzip.open(episodes / f"{i}.json.gz", "wt") as stream:
            json.dump(
                {
                    "resolved": i % 2 == 0,
                    "return": i,
                    "cost_usd": 0.5,
                    "events": [{"kind": "prepare"}, {"kind": "commit"}],
                    "termination": "SUBMITTED",
                    "transitions": [{"mask": [True, False, False]}, {"mask": [True, True, False]}],
                },
                stream,
            )
    live = weakref.WeakValueDictionary()
    peaks = []
    load = json.load

    class TrackedEpisode(dict):
        pass

    def tracked_load(stream):
        episode = TrackedEpisode(load(stream))
        live[id(episode)] = episode
        peaks.append(len(live))
        return episode

    monkeypatch.setattr(experiments.json, "load", tracked_load)
    result = experiments.report(tmp_path)
    assert max(peaks) <= 2  # the previous iteration can still exist while the next file is decoded
    assert result["episodes"] == result["genuine_choices"] == result["prepares"] == result["commits"] == 6
    assert result["resolved"] == result["cost_usd"] == 3
    assert result["success_rate"] == 0.5 and result["mean_return"] == 2.5
