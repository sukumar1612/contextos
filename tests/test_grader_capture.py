"""Complete grader capture must precede scoring; agent tool output remains bounded."""

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from contextos import grader, sandbox
from contextos.config import Config
from contextos.types import ToolResult

ROW = {
    "instance_id": "fixture__repo-1",
    "problem_statement": "Repair the fixture.",
    "repo": "fixture/repo",
    "base_commit": "a" * 40,
}


def local_grader(monkeypatch, script, *, resolved=True, override=None, patch_result=None):
    """A real local bash process substitutes for the container only in these tests."""
    boxes, parsed = [], []

    class LocalContainer:
        def __init__(self, cfg, task, name):
            self.cfg, self.closed = cfg.sandbox, False
            boxes.append(self)

        def start(self):
            pass

        def apply_patch(self, patch):
            return patch_result if patch_result is not None else ToolResult("")

        def run_script(self, body, timeout):
            if override is not None:
                return override
            return sandbox.run(["/bin/bash", "-s"], timeout, self.cfg.max_output_bytes, body.encode())

        def close(self):
            self.closed = True

    def bridge(cfg, operation, payload, directory):
        if operation == "script":
            return {"script": script, "marker": ">>>>> Applied Patch"}
        parsed.append(Path(payload["log"]).read_text())
        return {ROW["instance_id"]: {"resolved": resolved}}

    monkeypatch.setattr(grader, "container", LocalContainer)
    monkeypatch.setattr(grader, "bridge", bridge)
    return boxes, parsed


def test_grader_keeps_large_log_tail_without_uncapping_agent_tools(tmp_path, monkeypatch):
    command = "import sys; sys.stdout.write('x' * 1100000 + '\\nPASSED suite::test_repair\\n')"
    script = f"{shlex.quote(sys.executable)} -c {shlex.quote(command)}\n"
    boxes, parsed = local_grader(monkeypatch, script)
    cfg = Config()
    result = grader.grade(cfg, ROW, "patch", tmp_path / "verification")
    execution = json.loads((tmp_path / "verification" / "execution.json").read_text())
    assert result.resolved and result.status == "graded"
    assert len(parsed[0]) > 1_100_000 and "PASSED suite::test_repair" in parsed[0]
    assert "[OUTPUT CAPPED" not in parsed[0]
    assert cfg.sandbox.max_output_bytes == 1_000_000
    assert boxes[0].cfg is not cfg.sandbox and boxes[0].cfg.max_output_bytes is None
    assert boxes[0].closed
    assert execution["capture_version"] == grader.GRADER_CAPTURE_VERSION
    assert execution["capture_complete"] and execution["script_finished"]
    assert execution["returncode"] == execution["script_returncode"] == 0
    assert execution["captured_output_bytes"] > 1_100_000
    bounded = sandbox.run([sys.executable, "-c", command], 10, cfg.sandbox.max_output_bytes)
    assert "PASSED suite::test_repair" not in bounded.output
    assert "[OUTPUT CAPPED: " in bounded.output


@pytest.mark.parametrize("code", [124, 137, 143, -9, -15])
def test_interrupted_grading_is_persisted_but_not_parsed(tmp_path, monkeypatch, code):
    boxes, parsed = local_grader(monkeypatch, "", override=ToolResult("partial test output\n", code))
    with pytest.raises(RuntimeError, match="capture"):
        grader.grade(Config(), ROW, "patch", tmp_path)
    assert not parsed and boxes[0].closed
    assert "partial test output" in (tmp_path / ROW["instance_id"] / "test_output.txt").read_text()
    execution = json.loads((tmp_path / "execution.json").read_text())
    assert execution["returncode"] == code
    assert execution["interrupted"] and not execution["capture_complete"]


@pytest.mark.parametrize("code", [124, 137, 143])
def test_cleanup_cannot_hide_interrupted_test_command(tmp_path, monkeypatch, code):
    script = f"set -xo pipefail\n/bin/bash -c 'exit {code}'\ntrue\n"
    _, parsed = local_grader(monkeypatch, script)
    with pytest.raises(RuntimeError, match="capture"):
        grader.grade(Config(), ROW, "patch", tmp_path)
    execution = json.loads((tmp_path / "execution.json").read_text())
    assert execution["returncode"] == 0 and execution["script_finished"]
    assert code in execution["command_exit_codes"]
    assert execution["interrupted"] and not execution["capture_complete"]
    assert not parsed


@pytest.mark.parametrize("cleanup", ["", "true\n"])
def test_ordinary_failing_tests_remain_valid_failures(tmp_path, monkeypatch, cleanup):
    script = "set -xo pipefail\nprintf 'FAILED suite::test_repair\\n'\nfalse\n" + cleanup
    _, parsed = local_grader(monkeypatch, script, resolved=False)
    result = grader.grade(Config(), ROW, "patch", tmp_path)
    execution = json.loads((tmp_path / "execution.json").read_text())
    assert not result.resolved and result.status == "graded"
    assert parsed and "FAILED suite::test_repair" in parsed[0]
    assert execution["capture_complete"] and execution["script_finished"]
    assert execution["returncode"] == (0 if cleanup else 1)
    assert execution["command_exit_codes"] == [1]


def test_truncated_grading_output_is_not_parsed(tmp_path, monkeypatch):
    _, parsed = local_grader(
        monkeypatch, "", override=ToolResult("partial output\n[OUTPUT CAPPED: 2000000 bytes produced]")
    )
    with pytest.raises(RuntimeError, match="capture"):
        grader.grade(Config(), ROW, "patch", tmp_path)
    execution = json.loads((tmp_path / "execution.json").read_text())
    assert execution["output_capped"] and not execution["capture_complete"]
    assert not parsed


def test_missing_completion_marker_rejects_even_zero_exit_status(tmp_path, monkeypatch):
    _, parsed = local_grader(monkeypatch, "printf 'PASSED partial_test\\n'\nexit 0\n")
    with pytest.raises(RuntimeError, match="capture"):
        grader.grade(Config(), ROW, "patch", tmp_path)
    execution = json.loads((tmp_path / "execution.json").read_text())
    assert execution["returncode"] == 0 and not execution["script_finished"]
    assert not execution["capture_complete"] and not parsed


def test_uncapped_subprocess_timeout_preserves_all_partial_output(monkeypatch):
    partial = b"x" * 1_100_000 + b"\nlast captured line\n"

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1, output=partial)

    monkeypatch.setattr(sandbox.subprocess, "run", timeout)
    result = sandbox.run(["fixture"], 1, None)
    assert result.returncode == 124 and result.output.endswith("last captured line\n\n[TIMED OUT]")
    assert len(result.output) > 1_100_000


@pytest.mark.parametrize("code", [1, 124])
def test_patch_rejection_is_distinct_from_interrupted_patch_application(tmp_path, monkeypatch, code):
    boxes, parsed = local_grader(monkeypatch, "", patch_result=ToolResult("patch output", code))
    if code == 124:
        with pytest.raises(RuntimeError, match="patch application"):
            grader.grade(Config(), ROW, "patch", tmp_path)
    else:
        result = grader.grade(Config(), ROW, "patch", tmp_path)
        assert not result.resolved and result.status == "patch_rejected"
    execution = json.loads((tmp_path / "execution.json").read_text())
    assert execution["phase"] == "patch_application" and execution["patch_returncode"] == code
    assert execution["interrupted"] is (code == 124)
    assert (tmp_path / "patch_output.txt").read_text() == "patch output"
    assert not parsed and boxes[0].closed
