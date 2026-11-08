"""Local regression checks for the campaign wrapper; no Slurm, server, or model calls."""

import fcntl
import gzip
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from contextos.config import load_config

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "cluster" / "campaign.sbatch"


@pytest.fixture
def campaign_modules(monkeypatch):
    modules = []
    for name in ("run_state", "requeue_missing"):
        spec = importlib.util.spec_from_file_location(name, ROOT / "cluster" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


def published(directory, index, name=None):
    name = name or f"episode-{index}"
    folder = directory / "episodes"
    folder.mkdir(exist_ok=True)
    with gzip.open(folder / f"{name}.json.gz", "wt") as stream:
        json.dump({"name": name, "schedule_index": index}, stream)
    return name


def progress(directory, **overrides):
    state = {
        "strategy": "ppo",
        "training": True,
        "schedule_length": 10,
        "cursor": 0,
        "in_flight": [],
        "pending": [],
        "learned_names": [],
        "updates": [],
        "failed": [],
        "retry_attempts": [],
        **overrides,
    }
    (directory / "progress.json").write_text(json.dumps(state))
    return state


def shell_function(name):
    source = SCRIPT.read_text()
    # Load only the requested function, never the script's Slurm/server setup.
    return name + "() {" + source.split(name + "() {", 1)[1].split("\n}\n", 1)[0] + "\n}\n"


@pytest.mark.parametrize("training", ["0", "1"])
def test_common_sets_preserves_thinking_off_json_as_one_argument(training):
    script = (
        "set -u\nSERVED=$1; KW=$2; TRAIN=$3; HORIZON=200; REPEATS=4; "
        "MAX_EPISODE_SECONDS=3600; STRATEGY=ppo; TASKS=10\n"
        + shell_function("common_sets")
        + 'common_sets\nprintf "%s\\0" "${COMMON_SETS[@]}"\n'
    )
    result = subprocess.run(
        ["bash", "-c", script, "test", "qwen3.6-35b-a3b", '{"enable_thinking": false}', training],
        check=True,
        capture_output=True,
    )
    arguments = result.stdout.decode().rstrip("\0").split("\0")
    assert arguments[::2] == ["--set"] * (10 if training == "1" else 7)  # the encoder device is always set
    assert "encoder.device=cpu" in arguments  # the extracted function defaults it; the driver sets ENCODER_DEVICE
    assert 'provider.chat_template_kwargs={"enable_thinking": false}' in arguments
    config = load_config(overrides=arguments[1::2])
    assert config.provider.chat_template_kwargs == {"enable_thinking": False}
    assert config.experiment.max_interactions == 200
    assert config.learner.async_collection is (training == "1")


def test_common_sets_adds_the_checkpoint_and_extra_overrides():
    script = (
        "set -u\nSERVED=qwen3.6-35b-a3b; KW=null; TRAIN=0; HORIZON=200; REPEATS=2; MAX_EPISODE_SECONDS=3600; "
        "STRATEGY=ppo; CHECKPOINT=runs/full32-ppo-on-h200/policy.pt; "
        'EXTRA_SETS="encoder.semantic=false encoder.cache_size=0"\n'
        + shell_function("common_sets")
        + 'common_sets\nprintf "%s\\0" "${COMMON_SETS[@]}"\n'
    )
    result = subprocess.run(["bash", "-c", script], check=True, capture_output=True)
    arguments = result.stdout.decode().rstrip("\0").split("\0")
    assert arguments[::2] == ["--set"] * 10
    assert arguments[1::2][-3:] == [
        "strategy.checkpoint=runs/full32-ppo-on-h200/policy.pt",
        "encoder.semantic=false",
        "encoder.cache_size=0",
    ]
    config = load_config(overrides=arguments[1::2])
    assert config.strategy.checkpoint == "runs/full32-ppo-on-h200/policy.pt"
    assert config.encoder.semantic is False and config.encoder.cache_size == 0


def test_completion_requires_requested_schedule_positions(campaign_modules, tmp_path):
    state, _ = campaign_modules
    for index in (0, 2, 3):
        published(tmp_path, index)
    assert not state.complete(tmp_path, 3)  # three files do not fill positions 0, 1, 2
    published(tmp_path, 1)
    assert state.complete(tmp_path, 3)
    assert not state.complete(tmp_path, 5)
    assert state.complete(tmp_path, 0)
    with pytest.raises(ValueError, match="nonnegative"):
        state.complete(tmp_path, -1)


def test_training_completion_requires_committed_consumption(campaign_modules, tmp_path):
    state, _ = campaign_modules
    names = [published(tmp_path, index) for index in range(2)]
    assert state.complete(tmp_path, 2)
    assert not state.complete(tmp_path, 2, training=True)
    # The lightweight script tests checkpoint existence and its committed progress ledger.
    (tmp_path / "policy.pt").touch()
    progress(tmp_path, cursor=2, learned_names=names[:1], pending=names[1:])
    assert not state.complete(tmp_path, 2, training=True)
    progress(tmp_path, cursor=2, learned_names=names)
    assert state.complete(tmp_path, 2, training=True)  # legacy ledger remains supported
    progress(tmp_path, cursor=2, learned_names=names, consumed_names=[])
    assert not state.complete(tmp_path, 2, training=True)  # canonical ledger wins when present
    progress(tmp_path, cursor=2, learned_names=names, consumed_names=names)
    assert state.complete(tmp_path, 2, training=True)
    (tmp_path / "policy.pt").unlink()
    assert not state.complete(tmp_path, 2, training=True)


@pytest.mark.parametrize("index", [-1, True, "0"])
def test_invalid_schedule_index_is_not_complete(campaign_modules, tmp_path, index):
    state, _ = campaign_modules
    published(tmp_path, index, "invalid")
    with pytest.raises(ValueError, match="schedule index"):
        state.complete(tmp_path, 1)


def test_duplicate_schedule_index_is_not_complete(campaign_modules, tmp_path):
    state, _ = campaign_modules
    published(tmp_path, 0, "first")
    published(tmp_path, 0, "duplicate")
    with pytest.raises(ValueError, match="duplicate schedule index"):
        state.complete(tmp_path, 1)


def test_requeue_preserves_pending_commits_and_failures_outside_cap(campaign_modules, tmp_path):
    state, recovery = campaign_modules
    names = {i: published(tmp_path, i) for i in (0, 1, 3, 7)}
    failures = [
        {"name": "failed-2", "schedule_index": 2, "error": "retry me"},
        {"name": "failed-6", "schedule_index": 6, "error": "outside requested prefix"},
        {"name": "legacy-unknown", "error": "no trustworthy schedule index"},
    ]
    before = progress(
        tmp_path,
        cursor=8,
        in_flight=[0, 1, 6, 7],
        pending=[names[0], names[3], names[7], names[3]],
        learned_names=[names[0]],
        consumed_names=[names[0]],
        updates=[{"version": 1, "used_episodes": 1}],
        failed=failures,
        retry_attempts=[{"name": "earlier-retry"}],
    )
    episode_bytes = {path.name: path.read_bytes() for path in (tmp_path / "episodes").iterdir()}
    assert recovery.requeue(tmp_path, 4) == [2]
    after = json.loads((tmp_path / "progress.json").read_text())
    assert after["cursor"] == 8 and after["in_flight"] == [2, 6]
    assert after["pending"] == [names[3], names[7], names[1]]
    for key in ("learned_names", "consumed_names", "updates"):
        assert after[key] == before[key]
    assert after["failed"] == failures[1:]
    assert after["retry_attempts"] == [*before["retry_attempts"], failures[0]]
    assert {path.name: path.read_bytes() for path in (tmp_path / "episodes").iterdir()} == episode_bytes
    assert not state.complete(tmp_path, 4, training=True)


def test_requeue_full_launched_prefix_archives_legacy_failures(campaign_modules, tmp_path):
    _, recovery = campaign_modules
    failure = {"name": "old-failed-name", "error": "legacy failure"}
    progress(tmp_path, cursor=3, failed=[failure])
    assert recovery.requeue(tmp_path, 3) == [0, 1, 2]
    after = json.loads((tmp_path / "progress.json").read_text())
    assert after["failed"] == [] and after["retry_attempts"] == [failure]


def test_requeue_obeys_collector_lock(campaign_modules, tmp_path):
    _, recovery = campaign_modules
    progress(tmp_path, cursor=2)
    original = (tmp_path / "progress.json").read_bytes()
    with (tmp_path / ".run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another process owns"):
            recovery.requeue(tmp_path, 2)
    assert (tmp_path / "progress.json").read_bytes() == original
    assert recovery.requeue(tmp_path, 2) == [0, 1]


def test_requeue_rejects_cap_outside_schedule_without_mutation(campaign_modules, tmp_path):
    _, recovery = campaign_modules
    progress(tmp_path, cursor=2, schedule_length=4)
    original = (tmp_path / "progress.json").read_bytes()
    with pytest.raises(ValueError, match="schedule length"):
        recovery.requeue(tmp_path, 5)
    assert (tmp_path / "progress.json").read_bytes() == original


def test_campaign_restart_does_not_skip_published_unconsumed_training(tmp_path):
    # Execute the real phase loop with harmless local prefetch/server stubs. In particular,
    # no part of the script that reads credentials or runs squeue/sbatch is loaded.
    directory = tmp_path / "run-ppo-off-h200"
    directory.mkdir()
    names = [published(directory, index) for index in range(2)]
    progress(directory, cursor=2, pending=names)
    (directory / "policy.pt").touch()
    executable = tmp_path / ".venv" / "bin" / "contextos"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/bash\nexit 0\n")
    executable.chmod(0o700)
    source = SCRIPT.read_text()
    loop = "CAMPAIGN_FAILED=0" + source.split("CAMPAIGN_FAILED=0", 1)[1]
    script = (
        "set -uo pipefail\nAUDIT_ROOT=$1; AUDIT_PYTHON=$2; RUN=$3; AUDIT_MARKER=$4\n"
        'python3() { "$AUDIT_PYTHON" "$AUDIT_ROOT/$1" "${@:2}"; }\n'
        'log() { printf "%s\\n" "$*"; }\n'
        'ensure_server() { printf "needed\\n" > "$AUDIT_MARKER"; return 1; }\n'
        'done_episodes() { ls "$1"/episodes/*.json.gz 2>/dev/null | wc -l | tr -d " "; }\n'
        "TRAIN=1; COMMAND=train; PARTITIONS=train; STRATEGY=ppo; MODES=off; PHASES=200; "
        "MAX_EPISODES=2; TRIES=1; REPEATS=1; MAX_EPISODE_SECONDS=3600; "
        "SERVED=qwen3.6-35b-a3b; TASKS=; CONFIG_ARGS=()\n"
        + shell_function("run_complete")
        + shell_function("common_sets")
        + loop
    )
    marker = tmp_path / "server-needed"
    arguments = ["bash", "-c", script, "test", str(ROOT), sys.executable, str(tmp_path / "run"), str(marker)]
    unconsumed = subprocess.run(arguments, cwd=tmp_path, capture_output=True, text=True)
    assert unconsumed.returncode == 1, unconsumed.stderr
    assert marker.exists() and "remains incomplete" in unconsumed.stdout
    marker.unlink()
    progress(directory, cursor=2, consumed_names=names, learned_names=names)
    committed = subprocess.run(arguments, cwd=tmp_path, capture_output=True, text=True)
    assert committed.returncode == 0, committed.stderr
    assert not marker.exists() and "complete (2 episodes)" in committed.stdout
