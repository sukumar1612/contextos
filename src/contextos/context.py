"""PREPARE and COMMIT as two separate, asynchronous steps over the history."""

from __future__ import annotations

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor

from contextos.agent import SUMMARY_HEADINGS, SUMMARY_SYSTEM
from contextos.config import Config
from contextos.history import Group, History, Snapshot
from contextos.provider import EpisodeProvider
from contextos.tokens import Counter, api_messages
from contextos.types import Action, ContextLimit, Decision, SummaryInvalid


class ContextRuntime:
    def __init__(self, cfg: Config, history: History, provider, counter: Counter, tools: list[dict]):
        self.cfg, self.mode, self.horizon = cfg.context, cfg.context.mode, cfg.experiment.max_interactions
        self.history, self.provider, self.counter, self.tools = history, provider, counter, tools
        self.t = 0  # interactions completed
        self.last_commit = 0
        self.growth: list[int] = []  # size of each appended group
        self.latest_error = False
        self.job: dict | None = None  # {"snapshot", "summary", "cost"}; at most one
        self.future: Future | None = None
        self.jobs = 0
        self.summaries: list[tuple[int, float, str]] = []  # (source, cost, outcome)
        # outcome: "committed" | "unused" (ready, never committed: a scheduling choice)
        #          | "invalid" | "failed" (summarizer or service problem: cost, never waste)
        self.forced: dict[int, int] = {}  # source -> forced interventions
        self.events: list[dict] = []
        self.summary_jobs: list[dict] = []
        self.summary_wait_seconds = 0.0
        self.pool = ThreadPoolExecutor(max_workers=1)

    # ---- sizes and legality ----------------------------------------------------
    @property
    def phase(self) -> str:
        if self.job is None:
            return "IDLE"
        return "READY" if self.job["summary"] is not None else "PREPARING"

    def size(self) -> int:
        return self.counter.prompt(self.history.messages, self.tools)

    def eligible(self, emergency: bool = False) -> list[Group]:
        return self.history.prefix(self.counter, self.cfg.recent_tokens, emergency)

    def prefix_size(self, groups: list[Group] | None = None) -> int:
        groups = self.eligible() if groups is None else groups
        return self.counter.prompt([m for g in groups for m in g.messages], guarded=False) if groups else 0

    def candidate_size(self) -> int:
        if self.phase != "READY":
            return self.size()
        job = self._job()
        groups = self.history.replacement(job["snapshot"], job["summary"])
        return self.counter.prompt(self.history.pinned + [m for g in groups for m in g.messages], self.tools)

    def mask(self) -> list[bool]:
        prepare = self.phase == "IDLE" and self.prefix_size() >= self.cfg.minimum_prefix_tokens
        commit = False
        if self.phase == "READY":
            try:
                new = self.candidate_size()
                commit = new <= self.cfg.input_tokens and self.size() - new >= self.cfg.minimum_gain_tokens
            except SummaryInvalid:
                pass
        return [True, bool(prepare), bool(commit)]

    # ---- summarization, on the background thread ----------------------------------
    def _messages(self, groups: list[dict], partial: str) -> list[dict]:
        # Summarize what the model saw, not the raw replies: a thinking model's reply carries its reasoning
        # (thousands of tokens a step) which the API filter drops from prompts but json.dumps would keep,
        # inflating the records several-fold until no number of rounds fits them.
        records = [{**g, "messages": api_messages(g["messages"])} for g in groups]
        payload = {
            "task": self.history.pinned[1]["content"],
            "source_records": records,
            "previous_chunk_summary": partial,
        }
        return [
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    def _summarize(self, snapshot: Snapshot, job: dict) -> str:
        """Feed complete groups in chunks that fit; each round sees the previous partial summary."""
        pending, partial = list(snapshot.prefix), ""
        tags = (
            {"source": snapshot.source, "summary_job": snapshot.job}
            if isinstance(self.provider, EpisodeProvider)
            else {}
        )
        try:
            for _ in range(self.cfg.summary_rounds):
                chunk: list[dict] = []
                while (
                    pending
                    and self.counter.prompt(self._messages(chunk + [pending[0]], partial)) <= self.cfg.input_tokens
                ):
                    chunk.append(pending.pop(0))
                if not chunk:
                    raise SummaryInvalid("One complete source group cannot fit in a summary request")
                reply = self.provider.complete(
                    self._messages(chunk, partial), "summary", max_tokens=self.cfg.summary_output_tokens, **tags
                )
                job["cost"] += reply.cost_usd
                partial = reply.message.get("content") or ""
                if (
                    reply.finish_reason not in {"stop", "end_turn"}
                    or reply.message.get("tool_calls")
                    or any(h.lower() not in partial.lower() for h in SUMMARY_HEADINGS)
                ):
                    raise SummaryInvalid("Summary was empty, truncated, or missing required sections")
                if not pending:
                    return partial
            raise SummaryInvalid("Summary exceeded the maximum number of rounds")
        finally:
            if isinstance(self.provider, EpisodeProvider):
                job["cost"] = self.provider.job_cost(snapshot.job)  # includes failed/ambiguous requests
            job["metrics"].update(finished_at=time.time(), cost_usd=job["cost"])

    # ---- the transitions ----------------------------------------------------------
    def prepare(self, source: int, emergency: bool = False) -> None:
        if self.job:
            raise ValueError("Only one candidate summary at a time")
        prefix = self.eligible(emergency)
        if not prefix or (not emergency and self.prefix_size(prefix) < self.cfg.minimum_prefix_tokens):
            raise SummaryInvalid("No eligible prefix")
        job_id = f"j{self.jobs}"
        self.jobs += 1
        snapshot = self.history.snapshot(job_id, prefix, self.t, self.cfg.delay_interactions, source)
        metrics = {
            "job": job_id,
            "source": source,
            "prepare_interaction": self.t,
            "prepared_at": time.time(),
            "finished_at": None,
            "ready_at": None,
            "ready_interaction": None,
            "committed_at": None,
            "commit_interaction": None,
            "consumed_at": None,
            "consumed": False,
            "cost_usd": 0.0,
            "outcome": "preparing",
            "emergency": emergency,
        }
        self.summary_jobs.append(metrics)
        self.job = {"snapshot": snapshot, "summary": None, "cost": 0.0, "metrics": metrics}
        self.future = self.pool.submit(self._summarize, snapshot, self.job)
        self.events.append(
            {
                "kind": "prepare",
                "job": job_id,
                "source": source,
                "emergency": emergency,
                "t": self.t,
                "at": metrics["prepared_at"],
                "size": self.size(),
            }
        )

    def reconcile(self, force: bool = False, *, settling: bool = False) -> None:
        """Expose a finished summary. Controlled delay hides it until ready_after."""
        if self.job is None or self.phase == "READY":
            return
        job, future = self._job(), self._future()
        snapshot = job["snapshot"]
        visible = force or (future.done() if self.mode == "natural_async" else self.t >= snapshot.ready_after)
        if not visible:
            return
        waiting = not future.done()
        started = time.perf_counter()
        try:
            job["summary"] = future.result()
            # A finished but unusable summary must not occupy the only candidate slot forever.
            # replacement() also rejects a snapshot whose source generation no longer matches.
            candidate = self.candidate_size()
            if self.size() - candidate < self.cfg.minimum_gain_tokens:
                raise SummaryInvalid("Summary does not meet the minimum gain")
            job["metrics"].update(ready_at=time.time(), ready_interaction=self.t, outcome="ready")
            self.events.append(
                {
                    "kind": "ready",
                    "job": snapshot.job,
                    "t": self.t,
                    "forced_gate": force,
                    "at": self.job["metrics"]["ready_at"],
                }
            )
        except SummaryInvalid as exc:
            self.events.append({"kind": "summary_invalid", "job": snapshot.job, "reason": str(exc)})
            self._drop("invalid")
        except Exception as exc:  # transport or HTTP failure on the background path
            self.events.append(
                {"kind": "summary_failed", "job": snapshot.job, "error": f"{type(exc).__name__}: {exc}"[:300]}
            )
            self._drop("failed")
        finally:
            if waiting and not settling:
                self.summary_wait_seconds += time.perf_counter() - started
        self.future = None

    def commit(self, source: int, emergency: bool = False) -> None:
        if self.phase != "READY":
            raise SummaryInvalid("No ready candidate")
        new, old = self.candidate_size(), self.size()
        if new > self.cfg.input_tokens or old - new < self.cfg.minimum_gain_tokens:
            raise SummaryInvalid("Candidate does not fit or does not meet the minimum gain")
        job = self._job()
        snapshot = job["snapshot"]
        self.history.commit(snapshot, job["summary"])
        self.last_commit = self.t
        job["metrics"].update(
            committed_at=time.time(), commit_interaction=self.t, snapshot_age_interactions=self.t - snapshot.started
        )
        self.events.append(
            {
                "kind": "commit",
                "job": snapshot.job,
                "source": source,
                "size_before": old,
                "size_after": new,
                "generation": self.history.generation,
                "emergency": emergency,
                "t": self.t,
                "at": job["metrics"]["committed_at"],
            }
        )
        self._drop("committed")

    def consumed(self, source: int) -> None:
        """A commit is useful only after a subsequent coding request reads it."""
        present = {g.id for g in self.history.groups}
        for job in self.summary_jobs:
            if not job["consumed"] and f"summary:{job['job']}" in present:
                job.update(consumed=True, consumed_at=time.time(), consumed_source=source)
                self.events.append(
                    {
                        "kind": "summary_consumed",
                        "job": job["job"],
                        "source": source,
                        "t": self.t,
                        "at": job["consumed_at"],
                    }
                )

    def discard(self, reason: str) -> None:
        if self.job:
            self.events.append({"kind": "unused_summary", "job": self.job["snapshot"].job, "reason": reason})
            self._drop("unused")

    def _job(self) -> dict:
        assert self.job is not None, "no summary job in this phase"
        return self.job

    def _future(self) -> Future:
        assert self.future is not None, "no summary job in this phase"
        return self.future

    def _drop(self, outcome: str) -> None:
        job = self._job()
        self.summaries.append((job["snapshot"].source, job["cost"], outcome))
        job["metrics"].update(outcome=outcome, cost_usd=job["cost"])
        self.job = self.future = None

    # ---- what the runner calls ------------------------------------------------------
    def apply(self, decision: Decision, source: int) -> None:
        action = Action(decision.action)
        if not self.mask()[action]:
            raise ValueError("Strategy selected a masked action")
        if action == Action.PREPARE:
            self.prepare(source)
        elif action == Action.COMMIT:
            self.commit(source)

    def guard(self, source: int) -> None:
        """The prompt must fit: force the pending summary, then emergency summaries, then give up."""
        if self.size() <= self.cfg.input_tokens:
            return
        self.forced[source] = self.forced.get(source, 0) + 1
        self.events.append({"kind": "forced_intervention", "source": source})

        def use_pending() -> bool:
            if self.job:
                self.reconcile(force=True)
                if self.phase == "READY":
                    try:
                        self.commit(source, emergency=True)
                    except SummaryInvalid:
                        self.discard("oversized suffix or invalid source")
            return self.size() <= self.cfg.input_tokens

        if use_pending():
            return
        for _ in range(self.cfg.emergency_rebuilds):
            try:
                self.prepare(source, emergency=True)
            except SummaryInvalid as exc:
                raise ContextLimit(str(exc)) from exc
            if use_pending():
                return
        raise ContextLimit("Emergency compaction could not produce a fitting prompt")

    def settle(self) -> None:
        if self.job:
            self.reconcile(force=True, settling=True)
            self.discard("episode ended without commit")
        for job in self.summary_jobs:
            if job["outcome"] == "committed" and not job["consumed"]:
                job["outcome"] = "committed_unused"
                self.events.append(
                    {
                        "kind": "unused_summary",
                        "job": job["job"],
                        "reason": "committed but never consumed by a coding request",
                    }
                )

    def close(self) -> None:
        try:
            self.settle()  # exception paths must account for background work too
        finally:
            self.pool.shutdown(wait=True)
