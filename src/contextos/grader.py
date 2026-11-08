"""Grade a patch with the pinned SWE-Gym fork, in a fresh container."""

from __future__ import annotations

import json
import re
import subprocess
import uuid
from dataclasses import replace
from pathlib import Path

from contextos.config import Config
from contextos.data import public_task
from contextos.sandbox import container
from contextos.types import Grade

BRIDGE = Path(__file__).with_name("grader_bridge.py")
GRADER_CAPTURE_VERSION = 1


def capture_script(script: str, marker: str) -> str:
    """Mark actual completion and preserve command failures hidden by later cleanup commands."""
    return (
        "set -E\n"
        "trap 'contextos_command_status=$?; "
        f'printf "\\n{marker}_COMMAND_EXIT %s\\n" "$contextos_command_status"'
        + "' ERR\n"
        + script
        + "\ncontextos_script_status=$?\n"
        + f"printf '\\n{marker}_COMPLETE %s\\n' \"$contextos_script_status\"\n"
        + 'exit "$contextos_script_status"\n'
    )


def interrupted(returncode: int) -> bool:
    return returncode < 0 or returncode in {124, 137, 143}


def bridge(cfg: Config, operation: str, payload: dict, directory: Path) -> dict:
    """Run one operation inside the grader's own Python environment."""
    source, output = directory / f"{operation}-input.json", directory / f"{operation}-output.json"
    source.write_text(json.dumps(payload))
    subprocess.run(
        [
            cfg.benchmark.grader_python,
            str(BRIDGE),
            operation,
            str(Path(cfg.benchmark.grader_root).resolve()),
            str(source),
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return json.loads(output.read_text())


def grade(cfg: Config, row: dict, patch: str, directory: Path) -> Grade:
    if cfg.sandbox.kind == "fake":
        return Grade(True, "fake_fixture_only")
    directory.mkdir(parents=True, exist_ok=True)
    spec = bridge(cfg, "script", {"row": row}, directory)
    # Test parsers require the final pytest summary. Copy only the grader's config: mutating the
    # shared sandbox config would also remove the cap from concurrently running agent tools.
    grading_cfg = replace(cfg, sandbox=replace(cfg.sandbox, max_output_bytes=None))
    env = container(grading_cfg, public_task(row, cfg), directory.name + "-grader")
    try:
        env.start()
        applied = env.apply_patch(patch)
        if applied.returncode:
            (directory / "patch_output.txt").write_text(applied.output)
            (directory / "execution.json").write_text(
                json.dumps(
                    {
                        "capture_version": GRADER_CAPTURE_VERSION,
                        "phase": "patch_application",
                        "patch_returncode": applied.returncode,
                        "returncode": None,
                        "capture_complete": False,
                        "script_finished": False,
                        "interrupted": interrupted(applied.returncode),
                    },
                    indent=2,
                )
            )
            if interrupted(applied.returncode):
                raise RuntimeError("Grading patch application was interrupted; no verified outcome")
            return Grade(False, "patch_rejected")
        marker = "CONTEXTOS_GRADER_" + uuid.uuid4().hex
        output = env.run_script(capture_script(spec["script"], marker), cfg.sandbox.grader_timeout)
        # The fork reads the task ID from the log's parent directory name, lowercased.
        log = directory / row["instance_id"].lower() / "test_output.txt"
        log.parent.mkdir(exist_ok=True)
        log.write_text(spec["marker"] + " (pred)\n" + output.output)
        completed = re.findall(rf"(?m)^{marker}_COMPLETE (-?\d+)\r?$", output.output)
        command_codes = [int(code) for code in re.findall(rf"(?m)^{marker}_COMMAND_EXIT (-?\d+)\r?$", output.output)]
        capped = bool(re.search(r"(?m)^\[OUTPUT CAPPED: \d+ bytes produced\]\r?$", output.output))
        stopped = interrupted(output.returncode) or any(
            interrupted(code) for code in [*command_codes, *(int(code) for code in completed)]
        )
        finished = len(completed) == 1
        complete = finished and not capped and not stopped
        (directory / "execution.json").write_text(
            json.dumps(
                {
                    "capture_version": GRADER_CAPTURE_VERSION,
                    "phase": "test_script",
                    "patch_returncode": applied.returncode,
                    "returncode": output.returncode,
                    "script_returncode": int(completed[0]) if finished else None,
                    "command_exit_codes": command_codes,
                    "script_finished": finished,
                    "output_capped": capped,
                    "interrupted": stopped,
                    "capture_complete": complete,
                    "captured_output_bytes": len(output.output.encode("utf-8")),
                    "max_output_bytes": grading_cfg.sandbox.max_output_bytes,
                },
                indent=2,
            )
        )
        if not complete:
            raise RuntimeError("Grading capture was interrupted, truncated, or incomplete; inspect execution.json")
        report = bridge(cfg, "parse", {"row": row, "patch": patch, "log": str(log.resolve())}, directory)
        outcome = report[row["instance_id"]]
        return Grade(bool(outcome["resolved"]), "graded", outcome)
    finally:
        env.close()
