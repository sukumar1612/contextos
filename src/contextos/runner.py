"""One repair attempt, start to finish, in memory."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from contextos.agent import SYSTEM, TOOLS, error_messages, parse_actions
from contextos.config import Config
from contextos.context import ContextRuntime
from contextos.history import History
from contextos.observe import observe
from contextos.provider import EpisodeProvider
from contextos.sandbox import make_sandbox
from contextos.types import ContextLimit, Grade, PublicTask


def run_episode(
    cfg: Config,
    task: PublicTask,
    provider,
    counter,
    encoder,
    strategy,
    grade: Callable[[str], Grade],
    training: bool,
    name: str,
) -> dict:
    started, started_at = time.perf_counter(), time.time()
    agent_started = None
    timing = {
        "setup_seconds": None,
        "agent_seconds": None,
        "grading_seconds": None,
        "background_settle_seconds": 0.0,
        "summary_wait_seconds": 0.0,
        "observation_seconds": 0.0,  # includes tokenization and waiting for the shared encoder
    }
    provider = EpisodeProvider(provider, cfg, name)
    history = History([{"role": "system", "content": SYSTEM}, {"role": "user", "content": task.problem_statement}])
    runtime = ContextRuntime(cfg, history, provider, counter, TOOLS)
    env = make_sandbox(cfg, task, name)
    transitions: list[dict] = []
    costs: list[float] = []
    termination, format_errors = None, 0
    try:
        try:
            env.start()
            timing["setup_seconds"] = time.perf_counter() - started
            for phase, seconds in (getattr(env, "timings", None) or {}).items():
                timing[f"setup_{phase}_seconds"] = seconds  # pull, extract, clone, sanitize, lock_wait
            agent_started = time.perf_counter()
            while termination is None:
                # Boundary: the controller decides.
                if runtime.t >= cfg.experiment.max_interactions:
                    termination = "HORIZON"
                    break
                cap = cfg.experiment.max_episode_seconds
                if cap is not None and time.perf_counter() - agent_started >= cap:
                    termination = "TIME_LIMIT"  # wall-clock cap: graded like a horizon stop
                    break
                runtime.reconcile()
                observing = time.perf_counter()
                try:
                    observation, mask = observe(runtime, encoder), runtime.mask()
                finally:
                    timing["observation_seconds"] = (timing["observation_seconds"] or 0.0) + (
                        time.perf_counter() - observing
                    )
                if transitions:
                    transitions[-1].update(next_observation=observation.tolist(), next_mask=mask)
                decision = strategy.decide(observation, mask, training)
                index = len(transitions)
                transitions.append(
                    {
                        "observation": observation.tolist(),
                        "mask": mask,
                        "action": int(decision.action),
                        "log_prob": decision.log_prob,
                        "value": decision.value,
                        "version": decision.version,
                        "next_observation": [0.0] * len(observation),
                        "next_mask": [True, False, False],
                        "terminal": False,
                    }
                )
                costs.append(0.0)
                # Context: apply the decision, then make sure the prompt fits.
                try:
                    runtime.apply(decision, index)
                    runtime.guard(index)
                except ContextLimit:
                    termination = "CONTEXT_BUDGET_STOP"
                    break
                # Request: the agent acts.
                reply = provider.complete(runtime.history.messages, "foreground", TOOLS, step=runtime.t, source=index)
                runtime.consumed(index)
                costs[index] += reply.cost_usd
                transitions[index]["key"] = reply.key
                actions, error = parse_actions(reply)
                if error:
                    group, latest_error = error_messages(reply, error), True
                    format_errors += 1
                    if format_errors >= 3:
                        termination = "FORMAT_ERROR_STOP"
                else:
                    format_errors = 0
                    allowance = max(1, cfg.context.tool_output_tokens // len(actions))
                    results = [env.execute(a) for a in actions]
                    group = [
                        reply.message,
                        *(
                            {
                                "role": "tool",
                                "tool_call_id": a["call_id"],
                                "content": counter.truncate(r.output, allowance),
                            }
                            for a, r in zip(actions, results, strict=False)
                        ),
                    ]
                    latest_error = bool(results[-1].returncode)
                    if any(r.finished for r in results):
                        termination = "SUBMITTED"
                # Append: one whole group joins the history.
                appended = runtime.history.append(runtime.t, group)
                runtime.growth.append(counter.prompt(appended.messages, guarded=False))
                runtime.latest_error = latest_error
                runtime.t += 1
            timing["agent_seconds"] = time.perf_counter() - agent_started
            patch = env.patch()
        finally:
            if timing["setup_seconds"] is None:
                timing["setup_seconds"] = time.perf_counter() - started
            if agent_started is not None and timing["agent_seconds"] is None:
                timing["agent_seconds"] = time.perf_counter() - agent_started
            settling = time.perf_counter()
            try:
                runtime.close()  # settle every background charge, including on exception paths
            finally:
                timing["background_settle_seconds"] = time.perf_counter() - settling
                timing["summary_wait_seconds"] = runtime.summary_wait_seconds
                env.close()
        grading = time.perf_counter()
        try:
            verdict = grade(patch)
        finally:
            timing["grading_seconds"] = time.perf_counter() - grading
    except BaseException as exc:
        # The collector persists this for retries/skips, rather than treating
        # failed infrastructure attempts as free or as verified repair failures.
        timing["episode_seconds"] = time.perf_counter() - started
        exc.contextos_resources = {  # type: ignore[attr-defined]
            **provider.resources(),
            "task_id": task.instance_id,
            "timing": timing,
            "events": runtime.events,
            "summary_jobs": runtime.summary_jobs,
            "interactions": runtime.t,
            "started_at": started_at,
            "ended_at": time.time(),
        }
        raise
    timing["episode_seconds"] = time.perf_counter() - started
    result = finalize(cfg, task, transitions, costs, runtime, verdict, termination, patch)
    return {**result, "timing": timing, "started_at": started_at, "ended_at": time.time()}


def finalize(
    cfg: Config,
    task: PublicTask,
    transitions: list[dict],
    costs: list[float],
    runtime: ContextRuntime,
    verdict: Grade,
    termination: str,
    patch: str,
) -> dict:
    """Attribute every dollar to the decision that caused it, then score each transition."""
    resources = runtime.provider.resources() if isinstance(runtime.provider, EpisodeProvider) else {}
    waste = [0.0] * len(transitions)
    if resources:
        costs = [0.0] * len(transitions)
        for request in resources["requests"]:
            costs[request["source"]] += request["cost_usd"]
        for job in runtime.summary_jobs:
            if job["outcome"] in {"unused", "committed_unused"}:
                waste[job["source"]] += job["cost_usd"]
    else:  # direct callers using an untracked test provider
        for source, cost, outcome in runtime.summaries:
            costs[source] += cost
            if outcome == "unused":
                waste[source] += cost
    r = cfg.reward
    for i, t in enumerate(transitions):
        terminal = i == len(transitions) - 1
        facts: dict[str, Any] = {
            "cost_usd": costs[i],
            "forced": runtime.forced.get(i, 0),
            "waste_usd": waste[i],
            "resolved": verdict.resolved if terminal else None,
        }
        base = (
            (r.success if terminal and verdict.resolved else 0.0)
            - r.cost * facts["cost_usd"]
            - r.forced * facts["forced"]
            - r.waste * facts["waste_usd"]
        )
        # Potential shaping: Phi(s) = -weight * clipped prompt utilization.
        # Terminal Phi is zero, so discounted shaping telescopes exactly.
        phi = -r.potential * min(2.0, max(0.0, t["observation"][0]))
        next_phi = 0.0 if terminal else -r.potential * min(2.0, max(0.0, t["next_observation"][0]))
        shaping = cfg.learner.gamma * next_phi - phi
        t.update(facts=facts, terminal=terminal, base_reward=base, shaping=shaping, reward=base + shaping)
    return {
        "task_id": task.instance_id,
        "termination": termination,
        "resolved": verdict.resolved,
        "grade": asdict(verdict),
        "interactions": runtime.t,
        "cost_usd": sum(costs),
        "forced": sum(runtime.forced.values()),
        "return": sum(t["reward"] for t in transitions),
        "base_return": sum(t["base_reward"] for t in transitions),
        "choice_points": sum(sum(t["mask"]) > 1 for t in transitions),
        "events": runtime.events,
        "summary_jobs": runtime.summary_jobs,
        "transitions": transitions,
        "patch": patch,
        **resources,
    }
