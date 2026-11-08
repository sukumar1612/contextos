"""One container per attempt, on Docker or udocker. Agent commands never run on the host."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

from contextos.config import Config
from contextos.config import Sandbox as SandboxConfig
from contextos.types import PublicTask, ToolResult


@contextmanager
def image_lock(image: str):
    """Coordinate image start/eviction across local experiment processes."""
    import fcntl

    directory = Path(tempfile.gettempdir()) / f"contextos-images-{os.getuid()}"
    directory.mkdir(mode=0o700, exist_ok=True)
    with (directory / hashlib.sha256(image.encode()).hexdigest()).open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def template_name(image: str) -> str:
    """The once-extracted container every episode of an image is cloned from (node-wide, not per run)."""
    return "ctx-tpl-" + hashlib.sha256(image.encode()).hexdigest()[:12]


def _udocker_home() -> Path:
    return Path(os.environ.get("UDOCKER_DIR") or Path.home() / ".udocker")


def image_directory(image: str) -> Path:
    """Where udocker keeps a pulled image's manifest and blob links: <repos>/<repository>/<tag>, with the
    repository store at UDOCKER_REPOS when set (the shared layer cache) and under UDOCKER_DIR otherwise."""
    repos = Path(os.environ.get("UDOCKER_REPOS") or _udocker_home() / "repos")
    repository, tag = image, "latest"
    if ":" in image.rsplit("/", 1)[-1]:
        repository, _, tag = image.rpartition(":")
    return repos / repository / tag


def _blob_present(path: Path) -> bool:
    try:
        return path.stat().st_size > 0  # follows udocker's links into the layer store
    except OSError:
        return False


def image_complete(image: str) -> bool:
    """True when every blob the image's manifest names is in place. `udocker images` lists an image from the
    moment its pull starts, so an image another job is still pulling, or whose pull was cut off, looks present
    but cannot be instantiated ("create container: getting layers or json"); such an image must be pulled
    (under the shared lock, which waits for the running pull) rather than used."""
    directory = image_directory(image)
    try:
        manifest = json.loads((directory / "manifest").read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(manifest, dict):
        return False
    entries = manifest.get("layers") or manifest.get("fsLayers") or []
    digests = [str(entry.get("digest") or entry.get("blobSum") or "") for entry in entries if isinstance(entry, dict)]
    if not digests or not all(digests) or not all(_blob_present(directory / digest) for digest in digests):
        return False
    config = manifest.get("config")
    config_digest = config.get("digest") if isinstance(config, dict) else None
    if isinstance(config_digest, str) and config_digest:
        layers = Path(os.environ.get("UDOCKER_LAYERS") or _udocker_home() / "layers")
        return _blob_present(directory / config_digest) or _blob_present(layers / config_digest)
    return True


def pull_lock_directory() -> Path:
    """Where pull locks live. udocker's layer store is shared by every pull that writes to it: when the store
    is a shared filesystem (UDOCKER_REPOS / UDOCKER_LAYERS point at it, as the cluster jobs do), the lock
    must be shared too, or two jobs on different nodes race and leave a truncated image behind."""
    shared = os.environ.get("UDOCKER_REPOS") or os.environ.get("UDOCKER_LAYERS")
    if shared:
        return Path(shared).parent / "locks"
    return Path(tempfile.gettempdir()) / f"contextos-images-{os.getuid()}"


@contextmanager
def pull_lock():
    """One image pull at a time per layer store. Two concurrent pulls of images with common base layers race
    on the layer files (one renames the temporary download away while the other still needs it). Always
    taken after the image's own lock, never the other way round."""
    import fcntl

    directory = pull_lock_directory()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (directory / "pull.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


# Reset to the base commit and erase every other ref, remote, and reflog so the
# upstream fix cannot be recovered from Git history.
SANITIZE = r"""
set -euo pipefail
cd /testbed
git -c core.hooksPath=/dev/null reset --hard {base}
git -c core.hooksPath=/dev/null clean -ffd
git -c core.hooksPath=/dev/null checkout --detach {base}
git for-each-ref --format='delete %(refname)' | git update-ref --stdin
for remote in $(git remote); do git remote remove "$remote"; done
rm -f .git/ORIG_HEAD .git/FETCH_HEAD .git/MERGE_HEAD
rm -rf .git/logs
git reflog expire --expire=now --all
git gc --prune=now
test "$(git rev-parse HEAD)" = {base}
"""


def run(args: list[str], timeout: float, max_bytes: int | None, stdin: bytes | None = None) -> ToolResult:
    try:
        if stdin is not None:
            done = subprocess.run(args, input=stdin, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        else:
            done = subprocess.run(
                args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout
            )
    except subprocess.TimeoutExpired as exc:
        return ToolResult((exc.stdout or b"")[:max_bytes].decode("utf-8", "replace") + "\n[TIMED OUT]", 124)
    stdout = done.stdout
    output = stdout[:max_bytes].decode("utf-8", "replace")
    if max_bytes is not None and len(stdout) > max_bytes:
        output += f"\n[OUTPUT CAPPED: {len(stdout)} bytes produced]"
    return ToolResult(output, done.returncode)


def pull_with_retries(pull, retries: int, what: str) -> None:
    """Run `pull()` until it succeeds; a registry timeout should not cost an episode."""
    import time

    for attempt in range(retries + 1):
        try:
            pull()
            return
        except RuntimeError as exc:
            if attempt == retries:
                raise
            delay = 30 * (attempt + 1)
            logging.getLogger("contextos").warning(
                "pull of %s failed (%s); retrying in %ds", what, str(exc)[-120:], delay
            )
            time.sleep(delay)


def container_id(output: str) -> str:
    """`docker run -d` prints the ID last; warnings (for example a platform mismatch) may precede it."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines or not re.fullmatch(r"[0-9a-f]{12,64}", lines[-1]):
        raise RuntimeError(f"docker run did not return a container ID: {output[-300:]}")
    return lines[-1]


class Container:
    """What every backend shares: the agent protocol, patch export, and the grader's two commands."""

    cfg: SandboxConfig
    task: PublicTask
    name: str
    handle: str | None

    def _exec(self, script: str, timeout: int, stdin: bytes | None = None) -> ToolResult:
        raise NotImplementedError

    def shell(self, script: str, timeout: int | None = None, activate: bool = True) -> ToolResult:
        if activate and self.cfg.activation:
            script = self.cfg.activation + " && " + script
        return self._exec(script, timeout or self.cfg.tool_timeout)

    def execute(self, action: dict) -> ToolResult:
        if action["kind"] == "submit":
            return ToolResult("Submission accepted; this attempt is now closed.", finished=True)
        return self.shell(action["command"])

    def patch(self) -> str:
        base = shlex.quote(self.task.base_commit)
        result = self.shell(
            f"git -c core.hooksPath=/dev/null add -N -- . && git -c core.hooksPath=/dev/null diff --binary {base} -- .",
            activate=False,
        )
        if result.returncode or "[OUTPUT CAPPED" in result.output:
            raise RuntimeError("Patch export failed or was truncated")
        return result.output

    def apply_patch(self, patch: str) -> ToolResult:
        if patch == "":
            return ToolResult("Empty patch; workspace unchanged.")
        return self._exec("git apply --binary -", self.cfg.tool_timeout, patch.encode())

    def run_script(self, script: str, timeout: int) -> ToolResult:
        return self._exec("/bin/bash -s", timeout, script.encode())


class Docker(Container):
    def __init__(self, cfg: Config, task: PublicTask, name: str):
        prefix = f"{cfg.sandbox.namespace}-" if cfg.sandbox.namespace else ""
        self.cfg, self.task, self.name, self.handle = cfg.sandbox, task, "ctx-" + prefix + name, None

    def _docker(self, args: list[str], timeout: float = 180, stdin: bytes | None = None) -> ToolResult:
        return run(["docker", *args], timeout, self.cfg.max_output_bytes, stdin)

    def _checked(self, args: list[str], timeout: float = 180) -> str:
        result = self._docker(args, timeout)
        if result.returncode:
            raise RuntimeError(f"docker {args[0]} failed: {result.output[-2000:]}")
        return result.output

    def _pull_if_missing(self) -> bool:
        if not self._docker(["image", "inspect", self.task.image]).returncode:
            return False
        pull_with_retries(
            lambda: self._checked(["pull", self.task.image], timeout=1800), self.cfg.pull_retries, self.task.image
        )
        return True

    def ensure_image(self) -> bool:
        """Pull the image without starting anything (prefetch); True when a pull happened."""
        with image_lock(self.task.image):
            return self._pull_if_missing()

    def start(self) -> None:
        if self.cfg.minimum_free_gb:
            free = shutil.disk_usage(self.cfg.disk_path or ".").free / 2**30
            if free < self.cfg.minimum_free_gb:
                raise RuntimeError(
                    f"Disk admission stopped: {free:.1f} GiB free; requires {self.cfg.minimum_free_gb:.1f} GiB"
                )
        with image_lock(self.task.image):
            self._pull_if_missing()
            self._docker(["rm", "-f", self.name], timeout=60)  # only this run's stale container
            platform = ["--platform", self.cfg.platform] if self.cfg.platform else []
            self.handle = container_id(
                self._checked(
                    [
                        "run",
                        "-d",
                        *platform,
                        "--name",
                        self.name,
                        "--network",
                        "none",
                        "--cap-drop",
                        "ALL",
                        "--security-opt",
                        "no-new-privileges",
                        "--pids-limit",
                        str(self.cfg.pids_limit),
                        "--memory",
                        self.cfg.memory,
                        "--cpus",
                        str(self.cfg.cpus),
                        "--entrypoint",
                        "/bin/bash",
                        self.task.image,
                        "-c",
                        "trap 'exit 0' TERM; while :; do sleep 60; done",
                    ]
                )
            )
        result = self.shell(SANITIZE.format(base=shlex.quote(self.task.base_commit)), timeout=300, activate=False)
        if result.returncode:
            raise RuntimeError("Repository sanitation failed: " + result.output[-2000:])

    def _exec(self, script: str, timeout: int, stdin: bytes | None = None) -> ToolResult:
        timed = ["timeout", "--signal=TERM", "--kill-after=5", str(timeout), "/bin/bash", "-lc", script]
        assert self.handle is not None, "container not started"
        return self._docker(["exec", "-i", "--workdir", self.cfg.workdir, self.handle, *timed], timeout + 30, stdin)

    def close(self) -> None:
        if self.handle:
            self._docker(["rm", "-f", self.handle], timeout=60)
            self.handle = None


class Udocker(Container):
    """The same lifecycle on udocker: no daemon, no namespaces. A container is a directory, every command
    is a fresh process inside it, and changes persist between commands. There is no network isolation or
    resource cap: an unreachable proxy in the environment discourages egress, and the Slurm job limits CPU
    and memory. Honours UDOCKER_DIR from the environment (use local scratch on compute nodes)."""

    def __init__(self, cfg: Config, task: PublicTask, name: str):
        prefix = f"{cfg.sandbox.namespace}-" if cfg.sandbox.namespace else ""
        self.cfg, self.task, self.name, self.handle = cfg.sandbox, task, "ctx-" + prefix + name, None
        self.executable = cfg.sandbox.executable or "udocker"

    def _udocker(self, args: list[str], timeout: float = 180, stdin: bytes | None = None) -> ToolResult:
        return run([self.executable, "--quiet", *args], timeout, self.cfg.max_output_bytes, stdin)

    def _checked(self, args: list[str], timeout: float = 180) -> str:
        result = self._udocker(args, timeout)
        if result.returncode:
            raise RuntimeError(f"udocker {args[0]} failed: {result.output[-2000:]}")
        return result.output

    def _have_image(self) -> bool:
        listed = self._udocker(["images"], timeout=120).output
        present = any(line.split()[:1] == [self.task.image] for line in listed.splitlines())
        return present and image_complete(self.task.image)

    def _pull_if_missing(self) -> bool:
        if self._have_image():
            return False
        with pull_lock():
            if self._have_image():  # another job finished the same pull while we waited for the lock
                return False
            pull_with_retries(
                lambda: self._checked(["pull", self.task.image], timeout=1800), self.cfg.pull_retries, self.task.image
            )
        return True

    def ensure_image(self) -> bool:
        """Pull the image without creating a container (prefetch); True when a pull happened."""
        with image_lock(self.task.image):
            return self._pull_if_missing()

    def _have_container(self, name: str) -> bool:
        # `udocker ps` prints the NAMES column as a Python list literal, e.g. ['ctx-tpl-0123abcd4567'],
        # so match the (unique, hash-suffixed) name as a substring rather than as a whitespace token.
        listed = self._udocker(["ps"], timeout=120).output
        return any(name in line for line in listed.splitlines())

    def start(self) -> None:
        """Pull if needed, then build this episode's container by cloning a once-extracted template of the
        image (a copy on local disk is several times faster than extracting the layers again, and the
        grader's fresh container benefits too), and reset the repository. Phase times land in `timings`."""
        import time

        if self.cfg.minimum_free_gb:
            free = shutil.disk_usage(self.cfg.disk_path or ".").free / 2**30
            if free < self.cfg.minimum_free_gb:
                raise RuntimeError(
                    f"Disk admission stopped: {free:.1f} GiB free; requires {self.cfg.minimum_free_gb:.1f} GiB"
                )
        self.timings = {"lock_wait": 0.0, "pull": 0.0, "extract": 0.0, "clone": 0.0, "sanitize": 0.0}
        waiting = time.perf_counter()
        with image_lock(self.task.image):
            self.timings["lock_wait"] = time.perf_counter() - waiting
            started = time.perf_counter()
            if self._pull_if_missing():
                self.timings["pull"] = time.perf_counter() - started
            self._udocker(["rm", "-f", self.name], timeout=600)  # only this run's stale container
            mode = f"--execmode={self.cfg.udocker_mode}"
            if self.cfg.template_containers:
                template = template_name(self.task.image)
                if not self._have_container(template):
                    started = time.perf_counter()
                    created = self._udocker(
                        ["create", f"--name={template}", self.task.image], timeout=1800
                    )  # extracts the image
                    if created.returncode and "already exists" not in created.output:
                        raise RuntimeError(f"udocker create failed: {created.output[-2000:]}")
                    # F3 setup patches every ELF file of the image: minutes for the largest images when 40
                    # workers extract templates on one node at once; the episode cap bounds the whole setup.
                    self._checked(["setup", mode, template], timeout=1800)
                    self.timings["extract"] = time.perf_counter() - started
                started = time.perf_counter()
                self._checked(["clone", f"--name={self.name}", template], timeout=1800)
                self.timings["clone"] = time.perf_counter() - started
            else:
                started = time.perf_counter()
                self._checked(["create", f"--name={self.name}", self.task.image], timeout=1800)
                self.timings["extract"] = time.perf_counter() - started
            self._checked(["setup", mode, self.name], timeout=1800)
        self.handle = self.name
        started = time.perf_counter()
        result = self.shell(SANITIZE.format(base=shlex.quote(self.task.base_commit)), timeout=300, activate=False)
        self.timings["sanitize"] = time.perf_counter() - started
        if result.returncode:
            raise RuntimeError("Repository sanitation failed: " + result.output[-2000:])

    def _exec(self, script: str, timeout: int, stdin: bytes | None = None) -> ToolResult:
        timed = ["timeout", "--signal=TERM", "--kill-after=5", str(timeout), "/bin/bash", "-lc", script]
        env = [f"--env={k}={v}" for k, v in self.cfg.udocker_env.items()]
        return self._udocker(
            ["run", "--nobanner", f"--workdir={self.cfg.workdir}", *env, self.name, *timed], timeout + 30, stdin
        )

    def close(self) -> None:
        if self.handle:
            self._udocker(["rm", "-f", self.handle], timeout=600)  # deleting an extracted image takes a while
            self.handle = None


def evict_image(cfg: Config, image: str) -> None:
    """Remove an image no attempt will need again. Docker refuses while a container still uses it;
    udocker containers are separate copies, so its images can always go."""
    if cfg.sandbox.kind == "fake" or not cfg.sandbox.evict_images:
        return
    with image_lock(image):
        if cfg.sandbox.kind == "udocker":
            executable = cfg.sandbox.executable or "udocker"
            if cfg.sandbox.template_containers:
                run([executable, "--quiet", "rm", "-f", template_name(image)], 600, cfg.sandbox.max_output_bytes)
            result = run([executable, "--quiet", "rmi", image], 600, cfg.sandbox.max_output_bytes)
        else:
            result = run([cfg.sandbox.executable or "docker", "image", "rm", image], 120, cfg.sandbox.max_output_bytes)
    if result.returncode:
        logging.getLogger("contextos").info("image %s kept: %s", image, result.output.strip()[-120:])


def container(cfg: Config, task: PublicTask, name: str):
    """A real backend by kind; the grader uses this for its fresh verification container."""
    return (Udocker if cfg.sandbox.kind == "udocker" else Docker)(cfg, task, name)


class FakeSandbox:
    """Offline fixture: every command returns about 2.4 KB of text; the patch is fixed."""

    def __init__(self, cfg: Config, task: PublicTask, name: str):
        self.task = task

    def start(self) -> None:
        pass

    def execute(self, action: dict) -> ToolResult:
        if action["kind"] == "submit":
            return ToolResult("Submitted fixture.", finished=True)
        return ToolResult("fixture evidence " * 140 + f"\nCommand: {action['command']}")

    def patch(self) -> str:
        return "diff --git a/example.py b/example.py\n--- a/example.py\n+++ b/example.py\n@@ -1 +1 @@\n-bug\n+fixed\n"

    def close(self) -> None:
        pass


def make_sandbox(cfg: Config, task: PublicTask, name: str):
    return FakeSandbox(cfg, task, name) if cfg.sandbox.kind == "fake" else container(cfg, task, name)
