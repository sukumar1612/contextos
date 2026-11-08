"""SWE-Gym rows, a seeded split, and the public view of a task."""

from __future__ import annotations

import json
import random
from pathlib import Path

from contextos.config import Config
from contextos.types import PublicTask


def load_rows(cfg: Config) -> list[dict]:
    path = Path(cfg.benchmark.rows)
    if not path.exists():
        from datasets import load_dataset

        rows = list(load_dataset(cfg.benchmark.dataset, split="train"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sizes(ratios: dict[str, float], total: int) -> dict[str, int]:
    """Largest-remainder rounding of ratios over total; ties keep the given order."""
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to one")
    exact = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: int(value) for name, value in exact.items()}
    for name in sorted(ratios, key=lambda n: -(exact[n] - counts[n]))[: total - sum(counts.values())]:
        counts[name] += 1
    return counts


def split(cfg: Config, rows: list[dict]) -> dict[str, list[dict]]:
    """Seeded shuffle, then consecutive slices. Without explicit counts every row is used."""
    rows = sorted(rows, key=lambda r: r["instance_id"])
    random.Random(cfg.benchmark.seed).shuffle(rows)
    counts = cfg.benchmark.counts or sizes(cfg.benchmark.ratios, len(rows))
    parts, start = {}, 0
    for name, count in counts.items():
        parts[name] = rows[start : start + count]
        start += count
    if start > len(rows):
        raise ValueError(f"Split needs {start} tasks; only {len(rows)} available")
    return parts


def public_task(row: dict, cfg: Config) -> PublicTask:
    image_id = row["instance_id"].lower().replace("__", "_s_")  # SWE-Gym image naming
    return PublicTask(
        row["instance_id"],
        row["problem_statement"],
        row["repo"],
        row["base_commit"],
        cfg.benchmark.image_template.format(instance_id=image_id),
    )
