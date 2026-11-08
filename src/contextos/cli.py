"""contextos demo | train | evaluate | report | prefetch | probe"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from contextos.config import Config, load_config
from contextos.experiments import prefetch, report, run
from contextos.policy import LEARNED


def demo_config(output: Path, strategy: str) -> Config:
    """Byte counting, an 11,000-byte cap, a fake agent that submits on its ninth turn."""
    cfg = Config()
    cfg.provider.kind, cfg.sandbox.kind, cfg.encoder.semantic, cfg.strategy.name = "fake", "fake", False, strategy
    vars(cfg.context).update(
        input_tokens=11000,
        margin_tokens=0,
        recent_tokens=2000,
        minimum_prefix_tokens=2000,
        minimum_gain_tokens=500,
        tool_output_tokens=3000,
        summary_output_tokens=1500,
    )
    vars(cfg.experiment).update(max_interactions=12, attempts_per_task=4, evaluation_repeats=4, train_tasks=1)
    vars(cfg.learner).update(
        episodes_per_update=4, epochs=1, learning_starts=4, batch_size=8, hidden=[32, 32], target_interval=2
    )
    cfg.benchmark.rows, cfg.benchmark.counts = str(output / "rows.jsonl"), {"train": 2, "validation": 1, "test": 1}
    return cfg


def demo(output: Path, strategy: str, episodes: int) -> dict:
    cfg = demo_config(output, strategy)
    output.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "instance_id": f"fixture__repo-{i}",
            "repo": f"fixture/repo{i % 2}",
            "base_commit": "a" * 40,
            "problem_statement": f"Fix deterministic fixture issue {i}.",
        }
        for i in range(4)
    ]
    Path(cfg.benchmark.rows).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run(cfg, output / "run", strategy in LEARNED, "train", episodes)


def probe(cfg: Config) -> dict:
    """One tiny request to the configured server."""
    from contextos.provider import Provider
    from contextos.tokens import Counter

    provider = Provider(cfg, Counter(cfg.context, cfg.provider.reasoning_effort, cfg.provider.chat_template_kwargs))
    try:
        return provider.probe()
    finally:
        provider.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="contextos", description="Context scheduling for a fixed coding agent.")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_config(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("-c", "--config", action="append", default=[], help="YAML overlay; repeatable")
        p.add_argument("--set", action="append", default=[], help="section.key=value; repeatable")
        return p

    for name in ("train", "evaluate"):
        p = with_config(sub.add_parser(name))
        p.add_argument("--run", type=Path, required=True)
        p.add_argument("--max-episodes", type=int, help="run only the first N schedule entries; resume-safe")
        p.add_argument("--workers", type=int, help="concurrent episodes; overrides experiment.workers")
        if name == "evaluate":
            p.add_argument("--partition", default="test", choices=["validation", "test"])
    d = sub.add_parser("demo", help="offline fixture run; no server or container runtime needed")
    d.add_argument("--output", type=Path, default=Path("runs/demo"))
    d.add_argument("--strategy", default="smoothagent_lookahead")
    d.add_argument("--episodes", type=int, default=4)
    sub.add_parser("report").add_argument("--run", type=Path, required=True)
    p = with_config(sub.add_parser("prefetch", help="pull the images a schedule needs before running it"))
    p.add_argument("--partition", default="test", choices=["train", "validation", "test"])
    p.add_argument("--max-episodes", type=int, help="only the images of the first N schedule entries")
    with_config(sub.add_parser("probe", help="one tiny request to the configured server"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is noise
    if args.command == "demo":
        result = demo(args.output, args.strategy, args.episodes)
    elif args.command == "report":
        result = report(args.run)
    elif args.command == "prefetch":
        result = prefetch(
            load_config(args.config, args.set), args.partition, args.partition == "train", args.max_episodes
        )
    elif args.command == "probe":
        result = probe(load_config(args.config, args.set))
    else:
        training = args.command == "train"
        cfg = load_config(args.config, args.set)
        if args.workers:
            cfg.experiment.workers = args.workers
        try:
            result = run(
                cfg, args.run, training, "train" if training else args.partition, schedule_cap=args.max_episodes
            )
        except KeyboardInterrupt:
            print("interrupted; episodes that were in flight will run again on the next start", flush=True)
            os._exit(130)  # worker threads cannot be interrupted; their containers are reclaimed on restart
    print(json.dumps(result, indent=2, default=str))
