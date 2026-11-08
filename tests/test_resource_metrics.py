"""Resource accounting at asynchronous and failed-request boundaries."""

import gzip
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from contextos.cli import demo, demo_config
from contextos.context import ContextRuntime
from contextos.history import History
from contextos.observe import Encoder
from contextos.policy import make_strategy
from contextos.provider import EpisodeProvider, FakeProvider, Provider
from contextos.runner import run_episode
from contextos.tokens import Counter
from contextos.types import Action, Decision, Grade, PublicTask


def run_fixture(cfg, provider_type=FakeProvider, strategy=None):
    counter = Counter(cfg.context)
    task = PublicTask("fixture__repo-0", "Fix this fixture.", "fixture/repo", "a" * 40, "fixture")
    return run_episode(
        cfg,
        task,
        provider_type(cfg, counter),
        counter,
        Encoder(cfg),
        strategy or make_strategy(cfg),
        lambda patch: Grade(True, "fake_fixture_only"),
        False,
        "fixture",
    )


def test_all_requests_and_summary_costs_reconcile_to_rewards(tmp_path):
    demo(tmp_path, "late_lookahead", 1)
    path = next((tmp_path / "run" / "episodes").glob("*.json.gz"))
    with gzip.open(path, "rt") as stream:
        episode = json.load(stream)
    requests = episode["requests"]
    assert episode["usage"]["all"]["input_tokens"] == sum(r["input_tokens"] for r in requests)
    assert episode["usage"]["all"]["output_tokens"] == sum(r["output_tokens"] for r in requests)
    assert episode["usage"]["summary"]["input_tokens"] > 0
    assert episode["cost_usd"] == pytest.approx(sum(r["cost_usd"] for r in requests))
    assert episode["cost_usd"] == pytest.approx(sum(t["facts"]["cost_usd"] for t in episode["transitions"]))
    assert episode["reference_cost_usd"] == pytest.approx(episode["cost_usd"])
    assert episode["actual_cost_usd"] is None
    assert episode["summary_jobs"] and any(j["consumed"] for j in episode["summary_jobs"])
    for job in episode["summary_jobs"]:
        if job["consumed"]:
            assert job["prepared_at"] <= job["finished_at"] <= job["committed_at"] <= job["consumed_at"]
        assert job["cost_usd"] == pytest.approx(sum(r["cost_usd"] for r in requests if r["summary_job"] == job["job"]))
    assert all(value >= 0 for value in episode["timing"].values())


def test_background_summary_finishing_after_submission_is_counted(tmp_path):
    class PrepareOnFinalRequest:
        def decide(self, observation, mask, training):
            return Decision(Action.PREPARE if observation[2] >= 8 / 12 and mask[1] else Action.CONTINUE)

    class SlowSummary(FakeProvider):
        def complete(self, messages, purpose, tools=None, max_tokens=None, step=0):
            if purpose == "summary":
                time.sleep(0.04)
            return super().complete(messages, purpose, tools, max_tokens, step)

    cfg = demo_config(tmp_path, "late_lookahead")
    cfg.context.input_tokens = 100_000
    episode = run_fixture(cfg, SlowSummary, PrepareOnFinalRequest())
    assert episode["termination"] == "SUBMITTED"
    assert len(episode["summary_jobs"]) == 1
    job = episode["summary_jobs"][0]
    assert job["outcome"] == "unused" and not job["consumed"] and job["cost_usd"] > 0
    assert episode["timing"]["background_settle_seconds"] > 0.01
    assert episode["timing"]["summary_wait_seconds"] == 0
    assert episode["transitions"][-1]["facts"]["waste_usd"] == pytest.approx(job["cost_usd"])
    assert episode["cost_usd"] == pytest.approx(sum(r["cost_usd"] for r in episode["requests"]))


def test_committed_summary_is_unused_until_a_coding_request_reads_it(tmp_path):
    cfg = demo_config(tmp_path, "late_lookahead")
    cfg.context.input_tokens = 100_000
    counter = Counter(cfg.context)
    provider = EpisodeProvider(FakeProvider(cfg, counter), cfg, "fixture")
    history = History([{"role": "system", "content": "System"}, {"role": "user", "content": "Task"}])
    for interaction in range(4):
        history.append(interaction, [{"role": "user", "content": "evidence " * 400}])
    runtime = ContextRuntime(cfg, history, provider, counter, [])
    try:
        runtime.prepare(0)
        runtime.reconcile(force=True)
        runtime.commit(0)
        runtime.settle()
        assert runtime.summary_jobs[0]["outcome"] == "committed_unused"
        assert runtime.summary_jobs[0]["consumed"] is False
        assert len([e for e in runtime.events if e["kind"] == "unused_summary"]) == 1
        runtime.settle()  # finalization must be idempotent
        assert len([e for e in runtime.events if e["kind"] == "unused_summary"]) == 1
    finally:
        runtime.close()


def http_config(tmp_path, monkeypatch):
    cfg = demo_config(tmp_path, "late_lookahead")
    cfg.provider.kind, cfg.provider.api_key_env = "http", "FIXTURE_API_KEY"
    cfg.context.input_tokens = 100_000
    monkeypatch.setenv("FIXTURE_API_KEY", "fixture-secret")
    return cfg


def response():
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 2},
        },
    )


def test_ambiguous_retry_cost_is_retained_and_labelled(tmp_path, monkeypatch):
    cfg = http_config(tmp_path, monkeypatch)
    cfg.provider.retries = 1
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, text="unavailable") if len(calls) == 1 else response()

    provider = Provider(cfg, Counter(cfg.context), transport=httpx.MockTransport(handler))
    provider.sleep = lambda seconds: None
    tracked = EpisodeProvider(provider, cfg, "fixture")
    try:
        reply = tracked.complete([{"role": "user", "content": "hi"}], "foreground", max_tokens=8, source=0)
        data = tracked.resources()
        assert len(data["requests"]) == 2 and data["estimated_cost_usd"] > 0
        assert data["usage"]["all"] == {"input_tokens": 11, "output_tokens": 2}
        assert data["usage"]["unknown_usage_requests"] == 1
        assert data["cost_usd"] == pytest.approx(reply.cost_usd)
        assert data["reference_cost_usd"] == pytest.approx(
            (11 * cfg.provider.input_price + 2 * cfg.provider.output_price) / 1e6
        )
        assert data["actual_cost_usd"] is None
    finally:
        provider.close()


def test_provider_shares_concurrency_limit_across_coding_and_summaries(tmp_path, monkeypatch):
    cfg = http_config(tmp_path, monkeypatch)
    cfg.provider.max_concurrent_requests = 2
    active, maximum = 0, 0
    lock = threading.Lock()

    def handler(request):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.015)
        with lock:
            active -= 1
        return response()

    provider = Provider(cfg, Counter(cfg.context), transport=httpx.MockTransport(handler))
    tracked = EpisodeProvider(provider, cfg, "fixture")
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(
                    tracked.complete,
                    [{"role": "user", "content": "hi"}],
                    "summary" if i % 2 else "foreground",
                    max_tokens=8,
                    source=i,
                )
                for i in range(8)
            ]
            for future in futures:
                future.result()
        assert maximum == 2
        assert len(tracked.requests) == 8
    finally:
        provider.close()


def test_failed_episode_retains_earlier_usage_for_collector(tmp_path):
    class FailsSecondRequest(FakeProvider):
        def complete(self, messages, purpose, tools=None, max_tokens=None, step=0):
            if purpose == "foreground" and step == 1:
                raise RuntimeError("fixture transport failure")
            return super().complete(messages, purpose, tools, max_tokens, step)

    cfg = demo_config(tmp_path, "late_lookahead")
    with pytest.raises(RuntimeError, match="fixture transport failure") as error:
        run_fixture(cfg, FailsSecondRequest)
    data = error.value.contextos_resources
    assert data["interactions"] == 1 and data["cost_usd"] > 0
    assert data["usage"]["coding"]["input_tokens"] > 0
    assert data["timing"]["grading_seconds"] is None


@pytest.mark.parametrize("gamma", [1.0, 0.9])
def test_potential_shaping_has_zero_terminal_potential(tmp_path, gamma):
    cfg = demo_config(tmp_path, "late_lookahead")
    cfg.reward.potential, cfg.learner.gamma = 0.25, gamma
    episode = run_fixture(cfg)
    transitions = episode["transitions"]
    expected = cfg.reward.potential * transitions[0]["observation"][0]
    assert sum(gamma**i * t["shaping"] for i, t in enumerate(transitions)) == pytest.approx(expected)
    assert all(t["reward"] == pytest.approx(t["base_reward"] + t["shaping"]) for t in transitions)
    assert transitions[-1]["shaping"] == pytest.approx(cfg.reward.potential * transitions[-1]["observation"][0])


def test_summary_payload_carries_only_what_the_model_saw(tmp_path):
    """A thinking model's reply keeps its reasoning in the history record; the summarizer must not
    serialize it (it inflated the records several-fold and no number of rounds could fit them)."""
    from dataclasses import asdict

    cfg = demo_config(tmp_path, "late_lookahead")
    counter = Counter(cfg.context)
    provider = EpisodeProvider(FakeProvider(cfg, counter), cfg, "fixture")
    history = History([{"role": "system", "content": "System"}, {"role": "user", "content": "Task"}])
    history.append(
        0,
        [
            {
                "role": "assistant",
                "content": "Inspecting the parser",
                "reasoning": "hmm " * 5000,
                "reasoning_content": None,
            }
        ],
    )
    runtime = ContextRuntime(cfg, history, provider, counter, [])
    try:
        payload = runtime._messages([asdict(g) for g in history.groups], "")[-1]["content"]
        assert "Inspecting the parser" in payload and "hmm hmm" not in payload and "reasoning" not in payload
    finally:
        runtime.close()
