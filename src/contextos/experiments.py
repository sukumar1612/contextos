"""Training and evaluation runs: schedule, run directory, resume, image prefetch and the report."""

from __future__ import annotations

import collections
import copy
import gzip
import hashlib
import json
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path

from contextos.collector import (
    Collector,
    exclusive,
    load_progress,
    read_episode,
    rebuild_replay,
    reconcile_progress,
    write_json,
)
from contextos.config import Config
from contextos.data import load_rows, public_task, split
from contextos.observe import Encoder
from contextos.policy import Controller, make_strategy
from contextos.provider import FakeProvider, Provider
from contextos.tokens import Counter

__all__ = ["build_schedule", "exclusive", "prefetch", "read_episode", "report", "run", "write_json"]

log = logging.getLogger("contextos")

# Provider and run-control settings that may change when a paused run moves to another server or node.
_NON_IDENTITY = {
    "provider": (
        "base_url",
        "api_key_env",
        "api_key_file",
        "timeout_seconds",
        "retries",
        "max_concurrent_requests",
        "endpoint_file",
        "server_wait_seconds",
    ),
    "experiment": ("workers", "episode_retries", "max_consecutive_failures"),
    "encoder": ("device", "cache_size"),  # placement and caching do not change the feature definition
    "sandbox": ("namespace", "minimum_free_gb", "disk_path", "evict_images", "template_containers"),
}


def run(
    cfg: Config,
    directory: Path,
    training: bool,
    partition: str,
    max_episodes: int | None = None,
    max_seconds: float | None = None,
    schedule_cap: int | None = None,
) -> dict:
    directory = Path(directory)
    with exclusive(directory / ".run.lock"):
        return _run(copy.deepcopy(cfg), directory, training, partition, max_episodes, max_seconds, schedule_cap)


def build_schedule(cfg: Config, training: bool, partition: str) -> list[tuple[dict, int]]:
    """(row, repeat) pairs in run order: every task of the partition times its attempts. Excluded tasks leave
    their partition after the seeded split, so the other partitions keep their membership; prefixes
    (`train_tasks`, `evaluation_tasks`) then count eligible tasks only."""
    partitions = split(cfg, load_rows(cfg))
    excluded = set(cfg.experiment.exclude_tasks or [])
    unknown = excluded - {row["instance_id"] for rows in partitions.values() for row in rows}
    if unknown:
        raise ValueError(f"experiment.exclude_tasks names unknown tasks: {sorted(unknown)[:5]}")
    rows = [row for row in partitions[partition] if row["instance_id"] not in excluded]
    if not training and partition == "validation" and cfg.experiment.evaluation_tasks is not None:
        count = cfg.experiment.evaluation_tasks
        if type(count) is not int or count <= 0 or count > len(rows):
            raise ValueError("evaluation_tasks must be within the validation partition size")
        rows = rows[:count]
    if training and cfg.experiment.train_tasks:
        rows = rows[: cfg.experiment.train_tasks]
    repeats = cfg.experiment.attempts_per_task if training else cfg.experiment.evaluation_repeats
    return [(row, repeat) for row in rows for repeat in range(repeats)]


def prefetch(cfg: Config, partition: str, training: bool = False, schedule_cap: int | None = None) -> dict:
    """Pull every image the schedule (or its first schedule_cap entries) needs, one at a time, before any
    episode runs; the run itself then only extracts. Safe to run while a collector is up: pulls share locks."""
    from contextos.sandbox import container

    schedule = build_schedule(cfg, training, partition)
    if schedule_cap is not None:
        schedule = schedule[:schedule_cap]
    images: list[str] = []
    for row, _ in schedule:
        if row["instance_id"] not in images:
            images.append(row["instance_id"])
    if cfg.sandbox.kind == "fake":
        return {"images": len(images), "pulled": 0}
    pulled = 0
    for n, instance_id in enumerate(images, 1):
        row = next(r for r, _ in schedule if r["instance_id"] == instance_id)
        task = public_task(row, cfg)
        if container(cfg, task, "prefetch").ensure_image():
            pulled += 1
        log.info("prefetch %d/%d %s", n, len(images), task.image)
    return {"images": len(images), "pulled": pulled}


def fingerprint(cfg: Config, schedule: list[tuple[dict, int]]) -> str:
    """Identifies a run: every setting that changes its results, the dataset bytes and the schedule."""
    identity_config = asdict(cfg)
    for section, keys in _NON_IDENTITY.items():
        for key in keys:
            identity_config[section].pop(key, None)
    if cfg.strategy.checkpoint:
        identity_config["strategy"]["checkpoint"] = hashlib.sha256(
            Path(cfg.strategy.checkpoint).read_bytes()
        ).hexdigest()
    payload = {
        "config": identity_config,
        "dataset_sha256": hashlib.sha256(Path(cfg.benchmark.rows).read_bytes()).hexdigest(),
        "schedule": [(r["instance_id"], n) for r, n in schedule],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _run(
    cfg: Config,
    directory: Path,
    training: bool,
    partition: str,
    max_episodes: int | None = None,
    max_seconds: float | None = None,
    schedule_cap: int | None = None,
) -> dict:
    started = time.monotonic()
    if max_seconds is not None and (
        isinstance(max_seconds, bool)
        or not isinstance(max_seconds, int | float)
        or not math.isfinite(max_seconds)
        or max_seconds < 0
    ):
        raise ValueError("max_seconds must be a finite nonnegative number or None")
    deadline = None if max_seconds is None else started + max_seconds
    directory.mkdir(parents=True, exist_ok=True)
    cfg.sandbox.namespace = hashlib.sha256(str(directory.resolve()).encode()).hexdigest()[:10]
    (directory / "episodes").mkdir(exist_ok=True)
    schedule = build_schedule(cfg, training, partition)
    identity = {
        "strategy": cfg.strategy.name,
        "partition": partition,
        "training": training,
        "schedule_length": len(schedule),
        "fingerprint": fingerprint(cfg, schedule),
    }
    weights = directory / "policy.pt"
    if not (directory / "progress.json").exists() and (
        weights.exists() or any((directory / "episodes").glob("*.json.gz"))
    ):
        raise ValueError(f"{directory} has run artifacts but no progress.json to validate their identity")
    progress = load_progress(directory, identity)
    write_json(directory / "config.json", asdict(cfg))
    counter = Counter(cfg.context, cfg.provider.reasoning_effort, cfg.provider.chat_template_kwargs)
    encoder, strategy = Encoder(cfg), make_strategy(cfg)
    learning = training and isinstance(strategy, Controller)
    if (
        learning
        and not weights.exists()
        and (progress["updates"] or progress.get("consumed_names") or progress.get("learned_names"))
    ):
        raise FileNotFoundError(f"Cannot resume committed learner updates: {weights} is missing")
    if learning and weights.exists():
        strategy.load(weights, full=True)
        checkpoint_meta = strategy.checkpoint_metadata
        ledger = checkpoint_meta.get("consumed_names", checkpoint_meta.get("learned_names"))
        authoritative_ledger = ledger is not None
        if authoritative_ledger:
            progress["consumed_names"] = progress["learned_names"] = ledger
            progress["updates"] = checkpoint_meta["updates"]
        elif "consumed_names" not in progress and "learned_names" not in progress:
            # Compatibility with old checkpoints that predate the consumption ledger.
            # Their cursor included skipped work; replay reconstruction tolerates those missing files.
            excluded = set(progress["pending"]) | {f["name"] for f in progress["failed"]}
            progress["learned_names"] = [
                f"{row['instance_id']}-{repeat}-{cfg.strategy.name}"
                for i, (row, repeat) in enumerate(schedule[: progress["cursor"]])
                if i not in progress["in_flight"]
                and f"{row['instance_id']}-{repeat}-{cfg.strategy.name}" not in excluded
            ]
        if cfg.strategy.name == "double_dqn":
            names = progress.get("consumed_names", progress.get("learned_names", []))
            log.info(
                "replay buffer rebuilt from disk: %d transitions",
                rebuild_replay(directory, names, strategy, strict=authoritative_ledger),
            )
    if learning:
        progress["consumed_names"] = progress["learned_names"] = progress.get(
            "consumed_names", progress.get("learned_names", [])
        )
    reconcile_progress(directory, schedule, progress, learning)
    pending: list[dict] = []
    for name in progress["pending"]:  # finalized earlier, not yet learned from
        episode = read_episode(directory, name)
        if (
            learning
            and cfg.strategy.name != "double_dqn"
            and not cfg.learner.async_collection
            and any(t["version"] != strategy.version for t in episode["transitions"])
        ):
            raise ValueError(f"Pending on-policy episode {name} differs from saved policy version {strategy.version}")
        pending.append(episode)
    provider = FakeProvider(cfg, counter) if cfg.provider.kind == "fake" else Provider(cfg, counter)
    collector = Collector(
        cfg, directory, schedule, progress, provider, counter, encoder, strategy, training, pending, deadline
    )
    try:
        collector.schedule_cap = schedule_cap
        collector.collect(max_episodes)
        if learning:
            # Save even before the first optimizer step, and after async workers drain: choices and
            # actor RNG may have advanced since the latest learner checkpoint.
            collector.save_checkpoint(progress["consumed_names"], progress["updates"])
        collector.save_progress()  # persist checkpoint-ledger repairs even when no new episode was launched
    finally:
        provider.close()
    finished = progress["cursor"] == len(schedule) and not progress["in_flight"] and not progress["pending"]
    stop_reason = "schedule_complete" if finished else "time_limit" if collector.time_limit_reached else "episode_limit"
    return {
        **report(directory),
        "updates": len(progress["updates"]),
        "failed_episodes": len(progress["failed"]),
        "ran_now": collector.ran,
        "cursor": progress["cursor"],
        "in_flight": list(progress["in_flight"]),
        "finished": finished,
        "time_limit_reached": collector.time_limit_reached,
        "stop_reason": stop_reason,
    }


def report(directory: Path) -> dict:
    # Transition observations dominate episode files. Keep only one decoded episode in memory,
    # rather than materializing the entire training campaign for a small aggregate report.
    total = solved = genuine_choices = 0
    returns = cost = 0.0
    events: collections.Counter = collections.Counter()
    terminations: collections.Counter = collections.Counter()
    for path in sorted((directory / "episodes").glob("*.json.gz")):
        with gzip.open(path, "rt") as stream:
            episode = json.load(stream)
        total += 1
        solved += bool(episode["resolved"])
        returns += episode["return"]
        cost += episode["cost_usd"]
        events.update(e["kind"] for e in episode["events"])
        terminations.update([episode["termination"]])
        genuine_choices += sum(sum(t["mask"]) > 1 for t in episode["transitions"])
    if not total:
        return {"episodes": 0, "genuine_choices": 0}
    return {
        "episodes": total,
        "resolved": solved,
        "success_rate": solved / total,
        "mean_return": returns / total,
        "cost_usd": cost,
        "prepares": events["prepare"],
        "commits": events["commit"],
        "unused_summaries": events["unused_summary"],
        "failed_summaries": events["summary_failed"],
        "forced_interventions": events["forced_intervention"],
        "terminations": dict(terminations),
        "genuine_choices": genuine_choices,
    }
