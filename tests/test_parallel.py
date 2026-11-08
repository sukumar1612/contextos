"""24 concurrent episodes on the fake provider: both collector modes, retry and skip, resume of
in-flight episodes, and image eviction after a task's last attempt."""

import gzip
import json
from pathlib import Path

from contextos import collector
from contextos import provider as provider_module
from contextos.cli import demo_config
from contextos.experiments import run


def many_tasks(tmp_path, strategy, tasks=10, attempts=4, workers=24, **learner):
    cfg = demo_config(tmp_path, strategy)
    rows = [
        {
            "instance_id": f"fixture__repo-{i}",
            "repo": f"fixture/repo{i % 2}",
            "base_commit": "a" * 40,
            "problem_statement": f"Fix deterministic fixture issue {i}.",
        }
        for i in range(tasks + 2)
    ]
    tmp_path.mkdir(parents=True, exist_ok=True)
    Path(cfg.benchmark.rows).write_text("".join(json.dumps(r) + "\n" for r in rows))
    cfg.benchmark.counts = {"train": tasks, "validation": 1, "test": 1}
    cfg.experiment.train_tasks, cfg.experiment.attempts_per_task = None, attempts
    cfg.experiment.evaluation_repeats, cfg.experiment.workers = attempts, workers
    vars(cfg.learner).update(learner)
    return cfg


def progress(path):
    return json.loads((path / "run" / "progress.json").read_text())


def files(path):
    return sorted((path / "run" / "episodes").glob("*.json.gz"))


def patch_complete(monkeypatch, should_fail):
    original = provider_module.FakeProvider.complete

    def complete(self, messages, purpose, tools=None, max_tokens=None, step=0):
        should_fail(messages, purpose, step)
        return original(self, messages, purpose, tools, max_tokens, step)

    monkeypatch.setattr(provider_module.FakeProvider, "complete", complete)


def test_free_mode_double_dqn_with_24_workers(tmp_path):
    result = run(many_tasks(tmp_path, "double_dqn"), tmp_path / "run", True, "train")
    assert result["episodes"] == 40 and result["finished"] and result["failed_episodes"] == 0
    assert result["updates"] == 40  # one update per landed episode
    p = progress(tmp_path)
    assert p["cursor"] == 40 and p["in_flight"] == [] and p["pending"] == []


def test_batch_mode_ppo_with_24_workers(tmp_path):
    cfg = many_tasks(tmp_path, "ppo", episodes_per_update=24)
    result = run(cfg, tmp_path / "run", True, "train")
    assert result["episodes"] == 40 and result["finished"] and result["updates"] == 2  # 24 + 16
    saved = __import__("torch").load(tmp_path / "run" / "policy.pt", weights_only=False)
    assert saved["version"] == 2
    for file in files(tmp_path):  # every transition of the second batch was collected under version 1
        episode = json.load(gzip.open(file, "rt"))
        assert len({t["version"] for t in episode["transitions"]}) == 1


def test_parallel_retry_and_skip(tmp_path, monkeypatch):
    once = {"done": False}

    def should_fail(messages, purpose, step):
        task = messages[1]["content"]
        if purpose == "foreground" and step == 0 and "issue 3." in task and not once["done"]:
            once["done"] = True  # task 3: fails once, succeeds on retry
            raise RuntimeError("transient")
        if purpose == "foreground" and "issue 5." in task:
            raise RuntimeError("permanent")  # task 5: fails every time

    patch_complete(monkeypatch, should_fail)
    cfg = many_tasks(tmp_path, "double_dqn", attempts=1)
    cfg.experiment.max_consecutive_failures = 10
    result = run(cfg, tmp_path / "run", True, "train")
    assert result["episodes"] == 9 and result["failed_episodes"] == 1 and result["finished"]
    assert progress(tmp_path)["failed"][0]["name"].startswith("fixture__repo-5-")


def test_in_flight_episodes_rerun_after_a_crash(tmp_path):
    cfg = many_tasks(tmp_path, "late_lookahead")
    first = run(cfg, tmp_path / "run", False, "train", max_episodes=12)
    assert first["ran_now"] == 12 and progress(tmp_path)["cursor"] == 12
    from contextos.data import load_rows, split

    schedule = [(row, repeat) for row in split(cfg, load_rows(cfg))["train"] for repeat in range(4)]
    p = progress(tmp_path)  # pretend the process died with episodes 4 and 7 still running
    for index in (4, 7):
        row, repeat = schedule[index]
        (tmp_path / "run" / "episodes" / f"{row['instance_id']}-{repeat}-late_lookahead.json.gz").unlink()
    p["in_flight"] = [4, 7]
    (tmp_path / "run" / "progress.json").write_text(json.dumps(p))
    second = run(cfg, tmp_path / "run", False, "train")
    assert second["ran_now"] == 30 and len(files(tmp_path)) == 40 and second["finished"]
    assert progress(tmp_path)["in_flight"] == []


def test_images_evicted_once_per_task_after_its_last_attempt(tmp_path, monkeypatch):
    evicted = []
    monkeypatch.setattr(collector, "evict_image", lambda cfg, image: evicted.append(image))
    run(many_tasks(tmp_path, "smoothagent_lookahead"), tmp_path / "run", False, "train")
    assert len(evicted) == 10 and len(set(evicted)) == 10


def test_max_episodes_caps_the_schedule_position_across_resumes(tmp_path):
    cfg = many_tasks(tmp_path, "late_lookahead", workers=4)
    first = run(cfg, tmp_path / "run", False, "train", schedule_cap=8)
    assert first["ran_now"] == 8 and progress(tmp_path)["cursor"] == 8
    second = run(cfg, tmp_path / "run", False, "train", schedule_cap=8)  # same cap on resume: nothing more
    assert second["ran_now"] == 0 and progress(tmp_path)["cursor"] == 8
    third = run(cfg, tmp_path / "run", False, "train", schedule_cap=12)  # a higher cap continues to it
    assert third["ran_now"] == 4 and progress(tmp_path)["cursor"] == 12
