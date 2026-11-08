"""Soft launch deadlines preserve settled data and on-policy cohort boundaries."""

import gzip
import json
from concurrent.futures import Future
from pathlib import Path

import pytest

from contextos import collector, experiments
from contextos.cli import demo_config
from contextos.collector import Collector


def config(tmp_path, strategy="ppo", workers=1):
    cfg = demo_config(tmp_path, strategy)
    cfg.experiment.attempts_per_task = 8
    cfg.experiment.workers = workers
    cfg.learner.choice_only = True
    rows = [
        {
            "instance_id": f"fixture__repo-{i}",
            "repo": "fixture/repo",
            "base_commit": "a" * 40,
            "problem_statement": f"Fix fixture issue {i}.",
        }
        for i in range(4)
    ]
    Path(cfg.benchmark.rows).write_text("".join(json.dumps(row) + "\n" for row in rows))
    return cfg


def clock_per_episode(monkeypatch, seconds):
    clock = [0.0]
    attempt = collector.attempt

    def measured(*args, **kwargs):
        result = attempt(*args, **kwargs)
        clock[0] += seconds
        return result

    monkeypatch.setattr(experiments.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(collector, "attempt", measured)
    return clock


def progress(directory):
    return json.loads((directory / "progress.json").read_text())


def episodes(directory):
    results = []
    for path in (directory / "episodes").glob("*.json.gz"):
        with gzip.open(path, "rt") as stream:
            results.append(json.load(stream))
    return sorted(results, key=lambda e: e["schedule_index"])


@pytest.mark.parametrize("strategy", ["late_lookahead", "double_dqn"])
def test_soft_deadline_stops_new_free_mode_launches_and_drains(tmp_path, monkeypatch, strategy):
    cfg = config(tmp_path, strategy)
    clock_per_episode(monkeypatch, 0.6)
    result = experiments.run(cfg, tmp_path / "run", strategy == "double_dqn", "train", max_seconds=1.0)
    assert result["ran_now"] == result["episodes"] == 2
    assert result["time_limit_reached"] and result["stop_reason"] == "time_limit"
    assert not result["finished"] and result["in_flight"] == []


def test_ppo_flushes_settled_partial_cohort_at_deadline_without_mixing_versions(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg.learner.choices_per_update = 10_000
    clock_per_episode(monkeypatch, 0.6)
    directory = tmp_path / "run"
    first = experiments.run(cfg, directory, True, "train", max_seconds=1.0)
    assert first["episodes"] == 2 and first["updates"] == 1
    assert progress(directory)["updates"][-1]["deadline_flush"]
    assert progress(directory)["pending"] == []
    second = experiments.run(cfg, directory, True, "train", max_seconds=1.0)
    assert second["ran_now"] == 2 and second["updates"] == 2
    recorded = episodes(directory)
    assert [{t["version"] for t in e["transitions"]} for e in recorded] == [{0}, {0}, {1}, {1}]


def test_initialization_consumes_the_soft_time_budget(tmp_path, monkeypatch):
    cfg = config(tmp_path, "late_lookahead")
    clock = [0.0]
    original = experiments.Encoder

    def delayed_encoder(cfg):
        clock[0] += 2.0
        return original(cfg)

    monkeypatch.setattr(experiments.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(experiments, "Encoder", delayed_encoder)
    result = experiments.run(cfg, tmp_path / "run", False, "train", max_seconds=1.0)
    assert result["episodes"] == result["ran_now"] == result["cursor"] == 0
    assert result["time_limit_reached"] and result["stop_reason"] == "time_limit"


def test_batch_submission_never_queues_more_than_worker_capacity(tmp_path, monkeypatch):
    cfg = config(tmp_path, workers=2)
    cfg.learner.episodes_per_update = 8
    submitted, peaks = {}, []

    class DeferredExecutor:
        def __init__(self, max_workers, **kwargs):
            self.capacity = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, fn, *args):
            future = Future()
            submitted[future] = (fn, args)
            peaks.append(sum(not f.done() for f in submitted))
            assert peaks[-1] <= self.capacity
            return future

        def shutdown(self, **kwargs):
            pass

    def settle_one(futures, **kwargs):
        future = futures[0]
        fn, args = submitted[future]
        future.set_result(fn(*args))
        return {future}, set(futures[1:])

    monkeypatch.setattr(collector, "ThreadPoolExecutor", DeferredExecutor)
    monkeypatch.setattr(collector, "wait", settle_one)
    result = experiments.run(cfg, tmp_path / "run", True, "train")
    assert result["finished"] and result["episodes"] == 8 and result["updates"] == 1
    assert max(peaks) == 2
    assert all({t["version"] for t in e["transitions"]} == {0} for e in episodes(tmp_path / "run"))


def test_checkpoint_save_failure_keeps_pending_and_old_progress(tmp_path):
    cfg = config(tmp_path)
    pending = [{"name": "unlearned-episode"}]
    state = {
        "in_flight": [],
        "cursor": 0,
        "pending": ["unlearned-episode"],
        "learned_names": ["old-episode"],
        "updates": [{"version": 1}],
    }
    experiments.write_json(tmp_path / "progress.json", state)
    previous = (tmp_path / "progress.json").read_bytes()

    class SaveFailure:
        def update(self, episodes):
            return {"version": 2, "optimizer_steps": 1}

        def save(self, path, metadata):
            assert metadata["learned_names"] == ["old-episode", "unlearned-episode"]
            raise OSError("checkpoint write failed")

    collector = Collector(cfg, tmp_path, [], state, None, None, None, SaveFailure(), True, pending)
    with pytest.raises(OSError, match="checkpoint write failed"):
        collector.learn()
    assert collector.pending == pending
    assert state["learned_names"] == ["old-episode"] and state["updates"] == [{"version": 1}]
    assert (tmp_path / "progress.json").read_bytes() == previous


def test_interrupted_episode_publish_keeps_last_committed_inflight_cursor(tmp_path, monkeypatch):
    cfg = config(tmp_path, "late_lookahead")
    directory = tmp_path / "run"
    original = collector.os.replace

    def fail_episode(source, target):
        if str(target).endswith(".json.gz"):
            raise OSError("episode publish failed")
        return original(source, target)

    monkeypatch.setattr(collector.os, "replace", fail_episode)
    with pytest.raises(OSError, match="episode publish failed"):
        experiments.run(cfg, directory, False, "train", max_episodes=1)
    saved = progress(directory)
    assert saved["cursor"] == 1 and saved["in_flight"] == [0]
    assert not list((directory / "episodes").glob("*.json.gz"))
