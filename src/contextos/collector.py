"""The resumable, parallel episode collector and the run's on-disk state.

Episodes run on up to experiment.workers threads. Rules, evaluation and Double DQN keep every slot busy
and learn as episodes land ("free"); PPO either runs fork-join cohorts of episodes_per_update episodes
under one policy version ("batch") or, with learner.async_collection, keeps every slot busy and learns
from completed episodes as its episode/choice target is reached ("async"). progress.json holds the launch
cursor, the indices of episodes in flight, the names of finalized episodes not yet consumed, updates and
failures. Resume reconciles published episodes with the checkpoint's consumption ledger, then re-runs
unfinished in-flight work and continues.
A failed episode is retried, then skipped; several failures in a row abort the run.
"""

from __future__ import annotations

import collections
import gzip
import json
import logging
import os
import time
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from contextos.config import Config
from contextos.data import public_task
from contextos.grader import grade
from contextos.policy import Controller
from contextos.runner import run_episode
from contextos.sandbox import evict_image

log = logging.getLogger("contextos")


def write_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1))
    os.replace(tmp, path)


def read_episode(directory: Path, name: str) -> dict:
    with gzip.open(directory / "episodes" / f"{name}.json.gz", "rt") as stream:
        return json.load(stream)


def load_progress(directory: Path, identity: dict) -> dict:
    path = directory / "progress.json"
    if not path.exists():
        return {
            **identity,
            "cursor": 0,
            "in_flight": [],
            "pending": [],
            "updates": [],
            "failed": [],
            "retry_attempts": [],
        }
    progress = json.loads(path.read_text())
    if {k: progress.get(k) for k in identity} != identity:
        raise ValueError(f"{path} belongs to a different run; choose a new directory")
    progress.setdefault("in_flight", [])
    progress.setdefault("retry_attempts", [])
    return progress


def rebuild_replay(directory: Path, names: list[str], strategy: Controller, *, strict: bool = False) -> int:
    """Double DQN's buffer is the last replay_capacity transitions of learned episodes,
    in completion/update order. Those episodes are on disk, so the checkpoint does not carry the buffer.
    An episode has been learned from iff its policy version is older than the saved one, because
    Double DQN updates, and bumps the version, after every episode."""
    capacity, chunks, total = strategy.cfg.replay_capacity, [], 0
    for name in reversed(names):
        try:
            transitions = strategy.training_transitions(read_episode(directory, name))
        except FileNotFoundError:
            if strict:
                raise FileNotFoundError(f"Cannot rebuild replay: consumed episode {name} is missing") from None
            # Old checkpoints without an explicit consumption ledger included skipped episodes.
            continue
        if not transitions or transitions[0]["version"] >= strategy.version:
            continue
        chunks.append(transitions)
        total += len(transitions)
        if total >= capacity:
            break
    stream = [t for transitions in reversed(chunks) for t in transitions][-capacity:]
    strategy.replay = collections.deque((strategy.replay_item(t) for t in stream), maxlen=capacity)
    return len(stream)


def reconcile_progress(directory: Path, schedule: list, progress: dict, learning: bool) -> None:
    """Recover the gap between atomic episode publication and progress/checkpoint publication.

    The checkpoint's consumed names are authoritative. A published but unconsumed episode becomes
    pending; it must not be rerun and overwrite the replay experience. Missing unconsumed work is
    requeued. Recorded failures remain skipped unless explicitly put back in flight.
    """
    names = [f"{row['instance_id']}-{repeat}-{progress['strategy']}" for row, repeat in schedule]
    indices = {name: index for index, name in enumerate(names)}
    consumed = set(progress.get("consumed_names", progress.get("learned_names", [])))
    published = {name for name in names if (directory / "episodes" / f"{name}.json.gz").exists()}
    completed = published | consumed
    in_flight = list(dict.fromkeys(i for i in progress["in_flight"] if 0 <= i < len(schedule)))
    frontier = max(
        [progress["cursor"], *(indices[name] + 1 for name in completed if name in indices), *(i + 1 for i in in_flight)]
    )
    failed = {entry.get("name") for entry in progress["failed"]}
    explicit = set(in_flight)
    progress["cursor"] = min(frontier, len(schedule))
    # Keep eligible recovered holes ahead of later entries when a resumed run uses a schedule cap.
    progress["in_flight"] = sorted(
        i
        for i in set(in_flight) | set(range(progress["cursor"]))
        if names[i] not in completed and (i in explicit or names[i] not in failed)
    )
    if learning:
        pending = list(dict.fromkeys(name for name in progress["pending"] if name in published - consumed))
        recovered = sorted(
            published - consumed - set(pending),
            key=lambda name: ((directory / "episodes" / f"{name}.json.gz").stat().st_mtime_ns, indices[name]),
        )
        progress["pending"] = [*pending, *recovered]
    else:
        progress["pending"] = []


def attempt(
    cfg: Config, row: dict, name: str, directory: Path, provider, counter, encoder, strategy, training: bool
) -> tuple[dict | None, dict | None]:
    """Run one episode, retrying on failure. KeyboardInterrupt passes through."""

    def grader(patch: str):
        return grade(cfg, row, patch, directory / "verifier" / name)

    error: str | None = None
    failed_attempts: list[dict] = []
    for index in range(cfg.experiment.episode_retries + 1):
        try:
            episode = run_episode(
                cfg, public_task(row, cfg), provider, counter, encoder, strategy, grader, training, name
            )
            episode["retry_attempts"] = failed_attempts
            return episode, None
        except Exception as exc:  # one bad episode must not end a long run
            error = f"{type(exc).__name__}: {exc}"[:500]
            failed_attempts.append(
                {
                    "name": name,
                    "attempt": index + 1,
                    "error": error,
                    "resources": getattr(exc, "contextos_resources", None),
                }
            )
            log.warning("episode %s attempt %d failed: %s", name, index + 1, error)
    return None, {"error": error, "attempts": failed_attempts}


class Collector:
    schedule_cap: int | None = None  # set by run(): only schedule entries below this index are launched

    def __init__(
        self,
        cfg: Config,
        directory: Path,
        schedule: list,
        progress: dict,
        provider,
        counter,
        encoder,
        strategy,
        training: bool,
        pending: list[dict],
        deadline: float | None = None,
    ):
        self.cfg, self.directory, self.schedule, self.progress = cfg, directory, schedule, progress
        self.provider, self.counter, self.encoder, self.strategy, self.training = (
            provider,
            counter,
            encoder,
            strategy,
            training,
        )
        self.learning = training and isinstance(strategy, Controller)
        if self.learning and cfg.strategy.name == "ppo" and cfg.learner.async_collection:
            self.mode = "async"  # keep slots busy; learn when enough completed episodes/choices have settled
        else:
            self.mode = "batch" if self.learning and cfg.strategy.name == "ppo" else "free"
        self.pending = pending
        self.workers = max(1, cfg.experiment.workers)
        self.todo = collections.deque(
            [i for i in progress["in_flight"] if i < len(schedule)] + list(range(progress["cursor"], len(schedule)))
        )
        self.remaining = collections.Counter(schedule[i][0]["instance_id"] for i in self.todo)  # attempts left per task
        self.running: dict[Future, int] = {}
        self.ran = self.consecutive = 0
        self.stop: BaseException | None = None  # set to stop launching; raised once everything has drained
        self.executor: ThreadPoolExecutor | None = None
        self.deadline, self.time_limit_reached = deadline, False

    # ---- naming ------------------------------------------------------------------------------------
    def name(self, index: int) -> str:
        row, repeat = self.schedule[index]
        return f"{row['instance_id']}-{repeat}-{self.cfg.strategy.name}"

    def save_progress(self) -> None:
        self.progress["pending"] = [e["name"] for e in self.pending]
        write_json(self.directory / "progress.json", self.progress)

    def save_checkpoint(self, consumed: list[str], updates: list[dict]) -> None:
        # learned_names is the legacy name for the consumption ledger, including stale drops.
        # Per-update used_episodes/names distinguish actual PPO training data from consumed data.
        self.strategy.save(
            self.directory / "policy.pt",
            metadata={"consumed_names": consumed, "learned_names": consumed, "updates": updates},
        )

    def batch_ready(self) -> bool:
        """A nonzero choice target replaces the episode-count trigger in either PPO collection mode."""
        target = self.cfg.learner.choices_per_update
        if target:
            return sum(sum(sum(t["mask"]) > 1 for t in e["transitions"]) for e in self.pending) >= target
        return len(self.pending) >= self.cfg.learner.episodes_per_update

    # ---- one episode -------------------------------------------------------------------------------
    def launch(self, index: int) -> None:
        row, _ = self.schedule[index]
        assert self.executor is not None
        future = self.executor.submit(
            attempt,
            self.cfg,
            row,
            self.name(index),
            self.directory,
            self.provider,
            self.counter,
            self.encoder,
            self.strategy,
            self.training,
        )
        self.running[future] = index
        if index not in self.progress["in_flight"]:
            self.progress["in_flight"].append(index)
        self.progress["cursor"] = max(self.progress["cursor"], index + 1)
        self.ran += 1
        self.save_progress()

    def finish(self, future: Future, index: int) -> None:
        row, repeat = self.schedule[index]
        name = self.name(index)
        self.progress["in_flight"].remove(index)
        episode, error = future.result()
        if episode is None:
            self.consecutive += 1
            attempts = error.get("attempts", []) if isinstance(error, dict) else []
            self.progress["retry_attempts"].extend(attempts[:-1])
            failure = attempts[-1] if attempts else {"name": name, "error": error}
            failure["schedule_index"] = index
            self.progress["failed"].append(failure)
            if self.consecutive >= self.cfg.experiment.max_consecutive_failures:
                self.stop = self.stop or RuntimeError(
                    f"{self.consecutive} consecutive episodes failed; last error: {error}"
                )
        else:
            self.consecutive = 0
            self.progress["retry_attempts"].extend(episode.pop("retry_attempts", []))
            episode["name"], episode["repeat"], episode["schedule_index"] = name, repeat, index
            path = self.directory / "episodes" / f"{name}.json.gz"
            temporary = path.with_name(path.name + ".tmp")
            with gzip.open(temporary, "wt") as stream:
                json.dump(episode, stream)
            os.replace(temporary, path)
            if self.learning:
                self.pending.append(episode)
        self.remaining[row["instance_id"]] -= 1
        if self.remaining[row["instance_id"]] == 0:
            evict_image(self.cfg, public_task(row, self.cfg).image)
        self.save_progress()
        if self.mode == "free" and self.learning and self.pending:
            self.learn()  # Double DQN: after every episode, on whatever weights the workers are using
        elif self.mode == "async" and self.batch_ready():
            self.learn()  # async PPO: settled data reached the target while other episodes keep running

    def learn(self, *, deadline_flush: bool = False) -> None:
        started = time.perf_counter()
        metrics = self.strategy.update(self.pending)
        learner_seconds = time.perf_counter() - started  # excludes checkpoint and progress I/O
        consumed = [
            *self.progress.get("consumed_names", self.progress.get("learned_names", [])),
            *(e["name"] for e in self.pending),
        ]
        update = {
            "received_episodes": len(self.pending),
            **metrics,
            "consumed_episodes": len(consumed),
            "completed_episodes": len(consumed),
            "learner_seconds": learner_seconds,
            "time": time.time(),
        }
        if deadline_flush:
            update["deadline_flush"] = True
        updates = [*self.progress["updates"], update]
        # The checkpoint is the commit record. If saving fails or is interrupted,
        # retain the pending episodes and old progress. Resume already reconciles
        # checkpoint metadata if the process dies after this atomic save.
        self.save_checkpoint(consumed, updates)
        self.progress["consumed_names"] = self.progress["learned_names"] = consumed
        self.progress["updates"] = updates
        self.pending = []
        self.save_progress()

    # ---- the loops ----------------------------------------------------------------------------------
    def expired(self) -> bool:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.time_limit_reached = True
        return self.time_limit_reached

    def can_launch(self, max_episodes: int | None) -> bool:
        """max_episodes limits launches by this process (staged training asks for "N more now");
        schedule_cap limits the schedule position, so a resumed command-line run honours "the first N"."""
        return (
            not self.expired()
            and bool(self.todo)
            and self.stop is None
            and (max_episodes is None or self.ran < max_episodes)
            and (self.schedule_cap is None or self.todo[0] < self.schedule_cap)
        )

    def drain_one(self) -> None:
        done, _ = wait(list(self.running), timeout=1.0, return_when=FIRST_COMPLETED)  # timeout keeps Ctrl-C responsive
        for future in done:
            self.finish(future, self.running.pop(future))

    def collect_free(self, max_episodes: int | None) -> None:
        if self.learning and self.pending:
            self.learn()  # a crash can leave a finalized DDQN episode pending after the last launch
        while True:
            while self.can_launch(max_episodes) and len(self.running) < self.workers:
                self.launch(self.todo.popleft())
            if not self.running:
                return
            self.drain_one()

    def collect_async(self, max_episodes: int | None) -> None:
        """PPO without the fork-join wait: launch like free mode, learn when the settled-data target is met
        (see finish), and flush whatever is pending when the schedule or the deadline ends."""
        if self.pending and self.batch_ready():
            self.learn()
        while True:
            while self.can_launch(max_episodes) and len(self.running) < self.workers:
                self.launch(self.todo.popleft())
            if not self.running:
                break
            self.drain_one()
        if self.pending and self.stop is None:
            self.learn(deadline_flush=self.expired() and not self.batch_ready())

    def collect_batches(self, max_episodes: int | None) -> None:
        """Synchronous PPO: a cohort of episodes_per_update episodes runs under one frozen policy, every one
        of them finishes, then the policy updates."""
        size = self.cfg.learner.episodes_per_update
        target_choices = self.cfg.learner.choices_per_update
        while True:
            launched = 0
            wave_size = size if target_choices else max(0, size - len(self.pending))
            while True:
                # Bound submitted work as well as executing work: queued episodes
                # must not start after the soft deadline. The policy stays frozen
                # until this entire cohort has drained.
                while launched < wave_size and self.can_launch(max_episodes) and len(self.running) < self.workers:
                    self.launch(self.todo.popleft())
                    launched += 1
                if not self.running:
                    break
                self.drain_one()
            if self.stop is not None:
                return
            ready = self.batch_ready()
            expired = self.expired()
            if self.pending and (ready or not self.todo or expired):
                self.learn(deadline_flush=expired and not ready)
            if not self.can_launch(max_episodes):
                return

    def collect(self, max_episodes: int | None) -> None:
        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="episode") as executor:
            self.executor = executor
            try:
                {"batch": self.collect_batches, "async": self.collect_async, "free": self.collect_free}[self.mode](
                    max_episodes
                )
            except BaseException:
                # Keep the last atomically committed cursor/checkpoint. An
                # interrupt can arrive halfway through finish() or learn().
                # The outer process watchdog bounds non-cancellable workers.
                executor.shutdown(wait=False, cancel_futures=True)
                raise
        if self.stop is not None:
            raise self.stop


@contextmanager
def exclusive(path: Path) -> Iterator[None]:
    """POSIX advisory lock: fail clearly rather than run two writers on one state file."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process owns {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
