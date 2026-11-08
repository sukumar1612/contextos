import gzip
import json

import pytest

from contextos.cli import demo_config, main
from contextos.history import validate_protocol
from contextos.observe import observation_size
from contextos.types import validate_mask

STRATEGIES = ["smoothagent_lookahead", "late_lookahead", "ppo", "double_dqn"]


def episodes(path):
    return [json.load(gzip.open(p, "rt")) for p in sorted((path / "run" / "episodes").glob("*.json.gz"))]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_demo_runs_every_strategy(tmp_path, strategy, capsys):
    main(["demo", "--output", str(tmp_path), "--strategy", strategy, "--episodes", "4"])
    result = json.loads(capsys.readouterr().out)
    assert result["episodes"] == 4 and result["resolved"] == 4
    expected_size = observation_size(demo_config(tmp_path, strategy))
    assert expected_size == 524
    for episode in episodes(tmp_path):
        assert episode["termination"] == "SUBMITTED"
        assert episode["transitions"][-1]["terminal"]
        assert all(len(t["observation"]) == expected_size for t in episode["transitions"])
        assert all(t["mask"][0] and not (t["mask"][1] and t["mask"][2]) for t in episode["transitions"])
    if strategy in {"ppo", "double_dqn"}:
        assert result["updates"] >= 1 and (tmp_path / "run" / "policy.pt").exists()


def test_late_lookahead_is_forced_in_the_demo(tmp_path, capsys):
    main(["demo", "--output", str(tmp_path), "--strategy", "late_lookahead", "--episodes", "1"])
    episode = episodes(tmp_path)[0]
    kinds = [event["kind"] for event in episode["events"]]
    assert "prepare" in kinds and "forced_intervention" in kinds and "commit" in kinds
    forced = [t for t in episode["transitions"] if t["facts"]["forced"] and not t["terminal"]]
    assert forced and all(t["reward"] < -0.05 for t in forced)  # forced penalty plus cost
    assert episode["transitions"][-1]["reward"] > 9  # success minus cost


def test_ratio_split_uses_every_row():
    from contextos.config import Config
    from contextos.data import sizes, split

    assert sizes({"train": 0.75, "validation": 0.125, "test": 0.125}, 230) == {
        "train": 172,
        "validation": 29,
        "test": 29,
    }
    rows = [{"instance_id": f"repo__x-{i}"} for i in range(230)]
    parts = split(Config(), rows)  # default explicit counts
    assert {k: len(v) for k, v in parts.items()} == {"train": 170, "validation": 30, "test": 30}
    assert len({r["instance_id"] for part in parts.values() for r in part}) == 230
    cfg = Config()
    cfg.benchmark.counts = None  # ratios over every row
    assert {k: len(v) for k, v in split(cfg, rows).items()} == {"train": 172, "validation": 29, "test": 29}


def test_container_id_survives_docker_warnings():
    from contextos.sandbox import container_id

    warning = "WARNING: The requested image's platform (linux/amd64) does not match the host\n"
    assert container_id(warning + "33e0550b84b2e00d123e98121140bb11a3ff0b5832e50a129ca5b7a63a3601c1\n").startswith(
        "33e0550b"
    )
    with pytest.raises(RuntimeError):
        container_id(warning)


def test_protocol_and_mask_rules():
    validate_protocol([{"role": "assistant", "tool_calls": [{"id": "a"}]}, {"role": "tool", "tool_call_id": "a"}])
    with pytest.raises(ValueError):
        validate_protocol([{"role": "assistant", "tool_calls": [{"id": "a"}]}])
    with pytest.raises(ValueError):
        validate_mask([True, True, True])


def test_counter_reads_a_token_list_not_a_mapping():
    """transformers 5 returns {'input_ids', 'attention_mask'} from apply_chat_template by default;
    counting its length gave 2 tokens for every prompt and silently disabled the context cap."""
    from contextos.config import Config
    from contextos.tokens import Counter

    class Stub:
        def apply_chat_template(self, messages, **kw):
            ids = list(range(sum(len(m["content"]) for m in messages)))
            return ids if kw.get("return_dict") is False else {"input_ids": ids, "attention_mask": [1] * len(ids)}

    cfg = Config()
    cfg.context.margin_tokens = 0
    counter = Counter(cfg.context)
    counter.tokenizer = Stub()
    assert counter.prompt([{"role": "user", "content": "x" * 300}]) == 300
    assert counter.prompt([{"role": "user", "content": "x" * 700}]) == 700


def test_counter_measures_lists_without_a_user_turn():
    """Qwen3.8's template raises on a message list with no user query; partial histories are measured
    against a sentinel user message instead."""
    from contextos.config import Config
    from contextos.tokens import Counter

    class Strict:
        def apply_chat_template(self, messages, **kw):
            if not any(m["role"] == "user" for m in messages):
                raise ValueError("No user query found in messages.")
            return list(range(10 + sum(len(m.get("content") or "") for m in messages)))

    cfg = Config()
    cfg.context.margin_tokens = 0
    counter = Counter(cfg.context)
    counter.tokenizer = Strict()
    group = [
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "tool_call_id": "c", "content": "x" * 50},
    ]
    assert counter.prompt(group) == 50  # sentinel framing cancelled out
    assert counter.prompt([{"role": "user", "content": "y" * 7}] + group) == 67


def test_counter_passes_template_switches_to_the_template():
    """What the provider sends (reasoning effort, enable_thinking) is what the counter renders."""
    from contextos.config import Config
    from contextos.tokens import Counter

    seen = []

    class Stub:
        def apply_chat_template(self, messages, **kw):
            seen.append(kw)
            return list(range(10))

    cfg = Config()
    cfg.context.margin_tokens = 0
    counter = Counter(cfg.context, "low", {"enable_thinking": False})
    counter.tokenizer = Stub()
    assert counter.prompt([{"role": "user", "content": "hi"}]) == 10
    assert seen[-1]["enable_thinking"] is False and seen[-1]["reasoning_effort"] == "low"


def test_episode_wall_clock_cap_ends_the_episode_as_time_limit(tmp_path):
    """A capped episode stops at the next decision boundary, is graded, and is recorded as TIME_LIMIT."""
    import gzip
    import json
    from pathlib import Path

    from contextos.cli import demo_config
    from contextos.experiments import run

    cfg = demo_config(tmp_path, "late_lookahead")
    cfg.experiment.max_episode_seconds = 0.0
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
            for i in range(4)
        )
    )
    result = run(cfg, tmp_path / "run", False, "train", max_episodes=1)
    assert result["episodes"] == 1
    episode = json.load(gzip.open(next((tmp_path / "run" / "episodes").glob("*.json.gz")), "rt"))
    assert episode["termination"] == "TIME_LIMIT" and episode["interactions"] == 0
