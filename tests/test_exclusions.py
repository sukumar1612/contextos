"""Task exclusions remove tasks from their own partition after the seeded split and nothing else."""

import json
from pathlib import Path

import pytest

from contextos.cli import demo_config
from contextos.experiments import build_schedule


def rows(cfg, count=8):
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
            for i in range(count)
        )
    )
    cfg.benchmark.counts = {"train": 4, "validation": 2, "test": 2}
    cfg.experiment.train_tasks = None
    cfg.experiment.attempts_per_task, cfg.experiment.evaluation_repeats = 1, 2


def ids(schedule):
    return [row["instance_id"] for row, repeat in schedule if repeat == 0]


def test_excluding_a_task_keeps_every_other_partition_identical(tmp_path):
    cfg = demo_config(tmp_path, "late_lookahead")
    rows(cfg)
    before = {p: ids(build_schedule(cfg, p == "train", p)) for p in ("train", "validation", "test")}
    victim = before["validation"][0]
    cfg.experiment.exclude_tasks = [victim]
    after = {p: ids(build_schedule(cfg, p == "train", p)) for p in ("train", "validation", "test")}
    assert after["validation"] == before["validation"][1:]
    assert after["train"] == before["train"] and after["test"] == before["test"]
    assert len(build_schedule(cfg, False, "validation")) == 2  # one task left, two repeats


def test_prefixes_count_eligible_tasks_only(tmp_path):
    cfg = demo_config(tmp_path, "late_lookahead")
    rows(cfg)
    train = ids(build_schedule(cfg, True, "train"))
    cfg.experiment.exclude_tasks, cfg.experiment.train_tasks = [train[0]], 2
    assert ids(build_schedule(cfg, True, "train")) == train[1:3]


def test_unknown_exclusion_is_rejected(tmp_path):
    cfg = demo_config(tmp_path, "late_lookahead")
    rows(cfg)
    cfg.experiment.exclude_tasks = ["fixture__repo-99"]
    with pytest.raises(ValueError, match="unknown tasks"):
        build_schedule(cfg, True, "train")
