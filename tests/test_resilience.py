"""Failure handling and resumption: a bad episode is retried then skipped, a failed background summary
loses only its candidate, and restarting on the same directory continues from the cursor."""

import gzip
import json

import pytest
import torch

from contextos import provider as provider_module
from contextos.cli import demo, demo_config
from contextos.policy import Controller


def make_flaky(monkeypatch, should_fail):
    """Wrap FakeProvider.complete; should_fail(purpose, starts) decides, where starts counts episode starts."""
    original = provider_module.FakeProvider.complete
    state = {"starts": 0}

    def complete(self, messages, purpose, tools=None, max_tokens=None, step=0):
        if purpose == "foreground" and step == 0:
            state["starts"] += 1
        if should_fail(purpose, state["starts"]):
            raise RuntimeError(f"simulated {purpose} failure")
        return original(self, messages, purpose, tools, max_tokens, step)

    monkeypatch.setattr(provider_module.FakeProvider, "complete", complete)


def progress(path):
    return json.loads((path / "run" / "progress.json").read_text())


def episodes(path):
    return [json.load(gzip.open(p, "rt")) for p in sorted((path / "run" / "episodes").glob("*.json.gz"))]


def test_background_summary_failure_loses_only_the_candidate(tmp_path, monkeypatch):
    seen = {"summary": 0}

    def should_fail(purpose, starts):
        if purpose == "summary":
            seen["summary"] += 1
            return seen["summary"] == 1
        return False

    make_flaky(monkeypatch, should_fail)
    result = demo(tmp_path, "late_lookahead", 1)
    assert result["episodes"] == 1 and result["failed_episodes"] == 0 and result["failed_summaries"] == 1
    episode = episodes(tmp_path)[0]
    kinds = [event["kind"] for event in episode["events"]]
    assert "summary_failed" in kinds and "commit" in kinds and episode["termination"] == "SUBMITTED"
    events = episode["events"]
    failed_jobs = {e["job"] for e in events if e["kind"] == "summary_failed"}
    unused_jobs = {e["job"] for e in events if e["kind"] == "unused_summary"}
    assert failed_jobs and failed_jobs.isdisjoint(unused_jobs)  # a failed job is never recorded as unused
    unused_sources = {e["source"] for e in events if e["kind"] == "prepare" and e["job"] in unused_jobs}
    for index, t in enumerate(episode["transitions"]):  # waste only where a ready candidate went unused
        assert t["facts"]["waste_usd"] == 0 or index in unused_sources


def test_failed_episode_is_retried_then_skipped(tmp_path, monkeypatch):
    make_flaky(monkeypatch, lambda purpose, starts: purpose == "foreground" and starts in {2, 3})
    result = demo(tmp_path, "ppo", 4)  # second episode fails on its first try and its retry
    assert result["episodes"] == 3 and result["failed_episodes"] == 1 and result["finished"]
    assert result["updates"] == 1  # the three good episodes are learned from at the end of the schedule
    p = progress(tmp_path)
    assert p["cursor"] == 4 and p["pending"] == [] and len(p["failed"]) == 1


def test_consecutive_failures_abort_the_run(tmp_path, monkeypatch):
    make_flaky(monkeypatch, lambda purpose, starts: purpose == "foreground")
    with pytest.raises(RuntimeError, match="consecutive"):
        demo(tmp_path, "late_lookahead", 4)
    p = progress(tmp_path)
    assert p["cursor"] == 3 and len(p["failed"]) == 3


def test_restart_resumes_from_the_cursor(tmp_path):
    first = demo(tmp_path, "ppo", 2)
    assert first["ran_now"] == 2 and first["updates"] == 0 and not first["finished"]
    assert len(progress(tmp_path)["pending"]) == 2
    second = demo(tmp_path, "ppo", 4)
    assert second["ran_now"] == 2 and second["episodes"] == 4 and second["updates"] == 1 and second["finished"]
    saved = torch.load(tmp_path / "run" / "policy.pt", weights_only=False)
    assert saved["version"] == 1 and "optimizer" in saved and "replay" not in saved
    third = demo(tmp_path, "ppo", 4)
    assert third["ran_now"] == 0 and third["finished"] and third["updates"] == 1


def test_full_checkpoint_roundtrip(tmp_path):
    cfg = demo_config(tmp_path, "double_dqn")
    a = Controller(cfg)
    a.version, a.steps, a.choice_points, a.inserted = 3, 7, 11, 13
    a.save(tmp_path / "p.pt")
    saved = torch.load(tmp_path / "p.pt", weights_only=False)
    assert "replay" not in saved and (tmp_path / "p.pt").stat().st_size < 10_000_000
    b = Controller(cfg)
    b.load(tmp_path / "p.pt", full=True)
    assert (b.version, b.steps, b.choice_points, b.inserted) == (3, 7, 11, 13)
    c = Controller(cfg)
    c.load(tmp_path / "p.pt", full=False)
    assert c.version == 3


def test_resumed_dqn_run_is_identical_to_an_uninterrupted_one(tmp_path):
    straight, resumed = tmp_path / "straight", tmp_path / "resumed"
    demo(straight, "double_dqn", 4)
    demo(resumed, "double_dqn", 2)
    second = demo(resumed, "double_dqn", 4)
    assert second["ran_now"] == 2 and second["finished"]
    a = torch.load(straight / "run" / "policy.pt", weights_only=False)
    b = torch.load(resumed / "run" / "policy.pt", weights_only=False)
    assert a["version"] == b["version"] == 4 and a["inserted"] == b["inserted"]
    for key in a["network"]:
        assert torch.equal(a["network"][key], b["network"][key]), key  # bit-identical weights
    assert [e["return"] for e in episodes(straight)] == [e["return"] for e in episodes(resumed)]
