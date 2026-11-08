"""ContextOS snapshot, overlap, readiness and safe-commit invariants."""

import copy
import json
import threading

import pytest

from contextos.agent import SUMMARY_HEADINGS
from contextos.cli import demo_config
from contextos.context import ContextRuntime
from contextos.history import History, validate_protocol
from contextos.observe import Encoder
from contextos.provider import EpisodeProvider, FakeProvider
from contextos.runner import finalize, run_episode
from contextos.tokens import Counter
from contextos.types import Action, ContextLimit, Decision, Grade, PublicTask, Reply, SummaryInvalid, Usage


@pytest.fixture
def cfg(tmp_path):
    result = demo_config(tmp_path, "late_lookahead")
    result.context.input_tokens = 32768
    result.context.summary_output_tokens = 12288
    return result


def group(index, size=2500):
    return [
        {
            "role": "assistant",
            "content": f"Inspecting {index}",
            "tool_calls": [
                {
                    "id": f"c{index}",
                    "type": "function",
                    "function": {"name": "bash", "arguments": json.dumps({"command": f"inspect {index}"})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": f"c{index}", "content": f"Evidence {index}: " + "x" * size},
    ]


def history():
    result = History([{"role": "system", "content": "Pinned instructions"}, {"role": "user", "content": "Task"}])
    for index in range(4):
        result.append(index, group(index))
    return result


def runtime(cfg, provider_type=FakeProvider):
    counter = Counter(cfg.context)
    provider = provider_type(cfg, counter)
    result = ContextRuntime(cfg, history(), EpisodeProvider(provider, cfg, "fixture"), counter, [])
    result.t = 4
    return result, provider


class BlockingSummary(FakeProvider):
    def __init__(self, cfg, counter):
        super().__init__(cfg, counter)
        self.entered = threading.Event()
        self.release = threading.Event()

    def complete(self, messages, purpose, tools=None, max_tokens=None, step=0):
        if purpose == "summary":
            self.entered.set()
            if not self.release.wait(5):
                raise RuntimeError("test did not release the background summary")
        return super().complete(messages, purpose, tools, max_tokens, step)


@pytest.mark.parametrize("desired_gain", [-500, 0, 499])
def test_unusable_summary_releases_candidate_slot_without_erasing_cost_or_charging_waste(cfg, desired_gain):
    cfg.reward.waste = 1.0  # ensure a mistaken waste classification would change the reward
    base = "\n".join(f"{heading}\nKnown evidence." for heading in SUMMARY_HEADINGS)

    class FixedSummary(FakeProvider):
        content = base

        def complete(self, messages, purpose, tools=None, max_tokens=None, step=0):
            assert purpose == "summary" and self.counter.text(self.content) <= max_tokens
            self.calls += 1
            return Reply(
                {"role": "assistant", "content": self.content},
                Usage(self.counter.prompt(messages), self.counter.text(self.content)),
                "stop",
                2.5,
            )

    rt, provider = runtime(cfg, FixedSummary)
    try:
        source = rt.history.snapshot("sizing", rt.eligible(), rt.t, 2, 0)
        replacement = rt.history.replacement(source, base)
        base_size = rt.counter.prompt(rt.history.pinned + [m for g in replacement for m in g.messages], [])
        provider.content = base + "x" * (rt.size() - base_size - desired_gain)
        assert desired_gain < cfg.context.minimum_gain_tokens
        for index in range(2):
            assert rt.phase == "IDLE" and rt.mask()[1]  # invalid candidate cannot prevent another preparation
            rt.prepare(0)
            rt.future.result(timeout=5)
            rt.t += 2
            rt.reconcile()
            assert rt.phase == "IDLE" and rt.size() < cfg.context.input_tokens
            job = rt.summary_jobs[index]
            assert job["outcome"] == "invalid" and job["cost_usd"] == 2.5 and job["ready_at"] is None
        assert provider.calls == 2
        assert not any(e["kind"] in {"ready", "unused_summary"} for e in rt.events)
        transitions = [
            {
                "observation": [0.5],
                "next_observation": [0.5],
                "mask": [True, True, False],
                "action": int(Action.PREPARE),
            }
        ]
        task = PublicTask("fixture", "Task", "fixture/repo", "a" * 40, "fixture")
        episode = finalize(cfg, task, transitions, [0.0], rt, Grade(False, "fixture"), "HORIZON", "")
        assert episode["cost_usd"] == transitions[0]["facts"]["cost_usd"] == 5.0
        assert transitions[0]["facts"]["waste_usd"] == 0.0
        assert episode["return"] == -5.0
    finally:
        rt.close()


def test_snapshot_and_commit_preserve_pinned_and_every_suffix_group(cfg):
    rt, _ = runtime(cfg)
    try:
        pinned = copy.deepcopy(rt.history.pinned)
        rt.prepare(0)
        source = rt.job["snapshot"]
        function = rt.history.groups[0].messages[0]["tool_calls"][0]["function"]
        original = function["arguments"]
        function["arguments"] = "mutated outside the snapshot"
        assert source.prefix[0]["messages"][0]["tool_calls"][0]["function"]["arguments"] == original
        function["arguments"] = original
        for index in (4, 5):
            rt.history.append(index, group(index))
        suffix = copy.deepcopy(rt.history.groups[len(source.prefix) :])
        rt.future.result(timeout=5)
        rt.t = 6
        rt.reconcile()
        assert rt.phase == "READY" and rt.mask()[2]
        rt.commit(1)
        assert rt.phase == "IDLE" and rt.history.generation == 1
        assert rt.history.pinned == pinned and rt.history.groups[1:] == suffix
        assert rt.history.groups[0].id == "summary:j0"
        for item in rt.history.groups:
            validate_protocol(item.messages)
        with pytest.raises(SummaryInvalid, match="no longer matches"):
            rt.history.commit(source, "The same snapshot cannot be committed twice")
    finally:
        rt.close()


@pytest.mark.parametrize("mode,ready_after", [("controlled_delay", 2), ("natural_async", 1)])
def test_fast_summary_respects_readiness_mode_and_only_one_candidate(cfg, mode, ready_after):
    cfg.context.mode = mode
    rt, _ = runtime(cfg)
    try:
        rt.prepare(0)
        rt.future.result(timeout=5)  # completed work is still hidden until reconcile exposes it
        assert rt.phase == "PREPARING" and rt.mask() == [True, False, False]
        with pytest.raises(ValueError, match="Only one candidate"):
            rt.prepare(1)
        rt.t = 5
        rt.reconcile()
        assert rt.phase == ("READY" if ready_after == 1 else "PREPARING")
        if ready_after == 2:
            rt.t = 6
            rt.reconcile()
        assert rt.phase == "READY" and rt.mask() == [True, False, True]
        assert rt.summary_jobs[0]["ready_interaction"] == 4 + ready_after
        with pytest.raises(ValueError, match="Only one candidate"):
            rt.prepare(1)
        assert rt.jobs == 1
    finally:
        rt.close()


@pytest.mark.parametrize("mode", ["controlled_delay", "natural_async"])
def test_slow_summary_waits_only_at_controlled_gate(cfg, mode):
    cfg.context.mode = mode
    rt, provider = runtime(cfg, BlockingSummary)
    thread = None
    try:
        rt.prepare(0)
        assert provider.entered.wait(5)
        rt.t = 6  # the controlled readiness gate; natural readiness ignores this artificial delay
        called_result, completed = threading.Event(), threading.Event()
        future = rt.future
        original_result = future.result

        def marked_result(*args, **kwargs):
            called_result.set()
            return original_result(*args, **kwargs)

        future.result = marked_result
        errors = []

        def reconcile():
            try:
                rt.reconcile()
            except BaseException as exc:
                errors.append(exc)
            finally:
                completed.set()

        thread = threading.Thread(target=reconcile)
        thread.start()
        if mode == "controlled_delay":
            assert called_result.wait(5) and not completed.is_set()
        else:
            assert completed.wait(5) and not called_result.is_set()
            assert rt.phase == "PREPARING"
        provider.release.set()
        thread.join(timeout=5)
        assert not thread.is_alive() and not errors
        future.result(timeout=5)
        rt.reconcile()
        assert rt.phase == "READY"
        assert (rt.summary_wait_seconds > 0) == (mode == "controlled_delay")
    finally:
        provider.release.set()
        if thread is not None:
            thread.join(timeout=5)
        rt.close()


def test_stale_source_is_rejected_before_ready(cfg):
    rt, _ = runtime(cfg)
    try:
        rt.prepare(0)
        source = rt.job["snapshot"]
        rt.future.result(timeout=5)
        rt.history.commit(source, "Another compaction changed the source generation")
        rt.t = 6
        rt.reconcile()
        assert rt.phase == "IDLE" and rt.summary_jobs[0]["outcome"] == "invalid"
        assert rt.summary_jobs[0]["ready_at"] is None
        assert any(e["kind"] == "summary_invalid" and "no longer matches" in e["reason"] for e in rt.events)
    finally:
        rt.close()


def test_useful_summary_with_oversized_new_suffix_remains_a_guard_decision(cfg):
    cfg.context.emergency_rebuilds = 0  # isolate handling of the already prepared candidate
    rt, _ = runtime(cfg)
    try:
        rt.prepare(0)
        rt.future.result(timeout=5)
        rt.history.append(4, group(4, cfg.context.input_tokens))
        rt.t = 6
        rt.reconcile()
        assert rt.phase == "READY" and rt.candidate_size() > cfg.context.input_tokens
        assert rt.size() - rt.candidate_size() >= cfg.context.minimum_gain_tokens
        assert rt.mask() == [True, False, False]
        with pytest.raises(ContextLimit):
            rt.guard(1)
        assert rt.forced == {1: 1} and rt.phase == "IDLE"
        assert rt.summary_jobs[0]["outcome"] == "unused"
        assert not any(e["kind"] == "summary_invalid" for e in rt.events)
    finally:
        rt.close()


def test_prepare_overlaps_two_coding_requests_in_the_runner(cfg):
    cfg.context.input_tokens = 100_000
    requested = threading.Event()

    class PrepareOnce:
        def decide(self, observation, mask, training):
            if mask[1] and not requested.is_set():
                requested.set()
                return Decision(Action.PREPARE)
            return Decision(Action.CONTINUE)

    class Overlap(BlockingSummary):
        def __init__(self, cfg, counter):
            super().__init__(cfg, counter)
            self.overlapped_steps = []
            self.summary_threads = []

        def complete(self, messages, purpose, tools=None, max_tokens=None, step=0):
            if purpose == "summary":
                self.summary_threads.append(threading.get_ident())
                return super().complete(messages, purpose, tools, max_tokens, step)
            if requested.is_set() and not self.release.is_set():
                assert self.entered.wait(5)
                reply = FakeProvider.complete(self, messages, purpose, tools, max_tokens, step)
                self.overlapped_steps.append(step)
                if len(self.overlapped_steps) == 2:
                    self.release.set()
                return reply
            return FakeProvider.complete(self, messages, purpose, tools, max_tokens, step)

    counter = Counter(cfg.context)
    provider = Overlap(cfg, counter)
    task = PublicTask("fixture", "Fix this fixture.", "fixture/repo", "a" * 40, "fixture")
    try:
        episode = run_episode(
            cfg,
            task,
            provider,
            counter,
            Encoder(cfg),
            PrepareOnce(),
            lambda _: Grade(True, "fixture"),
            False,
            "fixture",
        )
    finally:
        provider.release.set()
    assert len(provider.overlapped_steps) == 2
    assert all(thread_id != threading.get_ident() for thread_id in provider.summary_threads)
    assert len(episode["summary_jobs"]) == 1 and episode["termination"] == "SUBMITTED"
