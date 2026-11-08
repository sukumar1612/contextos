"""The udocker backend issues the right command sequence. A stub executable stands in for udocker."""

import stat

from contextos import sandbox
from contextos.cli import demo_config
from contextos.sandbox import Udocker, evict_image
from contextos.types import PublicTask

STUB = r"""#!/bin/sh
# records every invocation; emulates: images (lists pulled images), pull, rm, create, setup, run, rmi
log="$STUB_LOG"; printf '%s' "$*" | tr '\n' ' ' >> "$log"; printf '\n' >> "$log"
shift   # --quiet
case "$1" in
  images) echo "REPOSITORY"; [ -f "$STUB_LOG.pulled" ] && cat "$STUB_LOG.pulled"; exit 0 ;;
  pull)   echo "$2" >> "$STUB_LOG.pulled"
          repo="$2"; tag=latest; case "${repo##*/}" in *:*) tag="${repo##*:}"; repo="${repo%:*}";; esac
          d="${UDOCKER_REPOS:-${UDOCKER_DIR:-$HOME/.udocker}/repos}/$repo/$tag"; mkdir -p "$d"
          printf '{"layers":[{"digest":"sha256:layer1"}],"config":{"digest":"sha256:config1"}}' > "$d/manifest"
          echo blob > "$d/sha256:layer1"; echo config > "$d/sha256:config1"; exit 0 ;;
  create) case "$*" in
            *ctx-tpl-*) name=$(echo "$*" | sed -n 's/.*--name=\([^ ]*\).*/\1/p')
                        echo "0123456789abcdef . W ['$name'] img" >> "$STUB_LOG.templates" ;;
          esac; echo "0123456789abcdef"; exit 0 ;;
  clone)  echo "fedcba9876543210"; exit 0 ;;
  ps)     echo "CONTAINER ID  P M NAMES IMAGE"; [ -f "$STUB_LOG.templates" ] && cat "$STUB_LOG.templates"; exit 0 ;;
  run)    echo "ran"; exit 0 ;;
  *)      exit 0 ;;
esac
"""


def test_udocker_command_sequence(tmp_path, monkeypatch):
    stub = tmp_path / "udocker"
    stub.write_text(STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("STUB_LOG", str(tmp_path / "calls.log"))
    monkeypatch.setenv("UDOCKER_DIR", str(tmp_path / "udocker-store"))
    monkeypatch.delenv("UDOCKER_REPOS", raising=False)
    monkeypatch.delenv("UDOCKER_LAYERS", raising=False)
    cfg = demo_config(tmp_path, "late_lookahead")
    cfg.sandbox.kind, cfg.sandbox.executable, cfg.sandbox.namespace = "udocker", str(stub), "run7"
    task = PublicTask("repo__x-1", "fix it", "repo/x", "a" * 40, "xingyaoww/sweb.eval.x86_64.repo_s_x-1:latest")
    box = Udocker(cfg, task, "repo__x-1-0-late_lookahead")
    box.start()
    assert box.execute({"kind": "bash", "command": "ls"}).output.strip() == "ran"
    assert box.execute({"kind": "submit"}).finished
    box.close()
    evict_image(cfg, task.image)
    calls = (tmp_path / "calls.log").read_text().splitlines()
    heads = [c.split()[1] for c in calls]
    # pull once, extract the image once into a template, clone the template for the episode, reset, run,
    # remove the episode container, and at eviction remove the template and the image
    assert heads == [
        "images",
        "images",
        "pull",
        "rm",
        "ps",
        "create",
        "setup",
        "clone",
        "setup",
        "run",
        "run",
        "rm",
        "rm",
        "rmi",
    ]
    template = sandbox.template_name(task.image)
    assert f"--name={template}" in calls[5] and "--execmode=F3" in calls[6] and calls[6].endswith(template)
    assert "--name=ctx-run7-repo__x-1-0-late_lookahead" in calls[7] and calls[7].endswith(template)
    assert "--execmode=F3" in calls[8] and calls[8].endswith("ctx-run7-repo__x-1-0-late_lookahead")
    assert "--workdir=/testbed" in calls[9] and "--env=https_proxy=http://127.0.0.1:9" in calls[9]
    assert "timeout --signal=TERM --kill-after=5" in calls[10] and calls[10].endswith("ls")
    assert "--nobanner" in calls[9] and calls[12].endswith(template) and calls[-1].endswith(task.image)
    assert set(box.timings) == {"lock_wait", "pull", "extract", "clone", "sanitize"} and box.timings["pull"] >= 0


def test_pulls_retry_before_failing(monkeypatch):
    import contextos.sandbox as sandbox

    monkeypatch.setattr(sandbox.time if hasattr(sandbox, "time") else __import__("time"), "sleep", lambda s: None)
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("curl: (28) Connection timeout")

    sandbox.pull_with_retries(flaky, retries=3, what="img")
    assert len(attempts) == 3
    with __import__("pytest").raises(RuntimeError):
        sandbox.pull_with_retries(lambda: (_ for _ in ()).throw(RuntimeError("down")), retries=1, what="img")


def test_pulls_are_serialised_per_machine(monkeypatch, tmp_path):
    """Four workers starting four different images must pull one at a time (shared udocker layer store)."""
    import tempfile
    import threading
    import time

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    active, peak = [], []

    def pull(image):
        with sandbox.image_lock(image), sandbox.pull_lock():
            active.append(image)
            peak.append(len(active))
            time.sleep(0.03)
            active.remove(image)

    threads = [threading.Thread(target=pull, args=(f"img{i}",)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(peak) == 4 and max(peak) == 1


def test_prefetch_pulls_each_scheduled_image_once(tmp_path, monkeypatch):
    """`contextos prefetch` pulls the images of the first N schedule entries before any episode starts."""
    import json
    from pathlib import Path

    from contextos.experiments import prefetch

    stub = tmp_path / "udocker"
    stub.write_text(STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("STUB_LOG", str(tmp_path / "calls.log"))
    monkeypatch.setenv("UDOCKER_DIR", str(tmp_path / "udocker-store"))
    monkeypatch.delenv("UDOCKER_REPOS", raising=False)
    monkeypatch.delenv("UDOCKER_LAYERS", raising=False)
    cfg = demo_config(tmp_path, "late_lookahead")
    cfg.sandbox.kind, cfg.sandbox.executable = "udocker", str(stub)
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
    first = prefetch(cfg, "train", False, schedule_cap=4)  # 2 train tasks x 4 repeats, capped to the first task
    again = prefetch(cfg, "train", False, schedule_cap=4)
    assert first == {"images": 1, "pulled": 1} and again == {"images": 1, "pulled": 0}
    calls = [c.split()[1] for c in (tmp_path / "calls.log").read_text().splitlines()]
    assert calls == ["images", "images", "pull", "images"]  # the second listing re-checks under the pull lock


def test_udocker_without_templates_extracts_per_episode(tmp_path, monkeypatch):
    stub = tmp_path / "udocker"
    stub.write_text(STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("STUB_LOG", str(tmp_path / "calls.log"))
    monkeypatch.setenv("UDOCKER_DIR", str(tmp_path / "udocker-store"))
    monkeypatch.delenv("UDOCKER_REPOS", raising=False)
    monkeypatch.delenv("UDOCKER_LAYERS", raising=False)
    cfg = demo_config(tmp_path, "late_lookahead")
    cfg.sandbox.kind, cfg.sandbox.executable, cfg.sandbox.template_containers = "udocker", str(stub), False
    task = PublicTask("repo__x-1", "fix it", "repo/x", "a" * 40, "xingyaoww/sweb.eval.x86_64.repo_s_x-1:latest")
    box = Udocker(cfg, task, "x")
    box.start()
    box.close()
    heads = [c.split()[1] for c in (tmp_path / "calls.log").read_text().splitlines()]
    assert heads == ["images", "images", "pull", "rm", "create", "setup", "run", "rm"] and box.timings["clone"] == 0.0


def test_second_episode_of_an_image_clones_the_existing_template(tmp_path, monkeypatch):
    """`udocker ps` lists names as a Python list literal; the template must be recognised, not re-created."""
    stub = tmp_path / "udocker"
    stub.write_text(STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("STUB_LOG", str(tmp_path / "calls.log"))
    monkeypatch.setenv("UDOCKER_DIR", str(tmp_path / "udocker-store"))
    monkeypatch.delenv("UDOCKER_REPOS", raising=False)
    monkeypatch.delenv("UDOCKER_LAYERS", raising=False)
    cfg = demo_config(tmp_path, "late_lookahead")
    cfg.sandbox.kind, cfg.sandbox.executable = "udocker", str(stub)
    task = PublicTask("repo__x-1", "fix it", "repo/x", "a" * 40, "xingyaoww/sweb.eval.x86_64.repo_s_x-1:latest")
    for name in ("x-0", "x-1"):
        box = Udocker(cfg, task, name)
        box.start()
        box.close()
    heads = [c.split()[1] for c in (tmp_path / "calls.log").read_text().splitlines()]
    assert heads.count("create") == 1 and heads.count("clone") == 2


def test_pull_lock_lives_in_the_shared_layer_store(tmp_path, monkeypatch):
    """With a shared udocker layer store (UDOCKER_REPOS), the pull lock is shared too, so jobs on different
    nodes cannot pull into the same store at once."""
    monkeypatch.setenv("UDOCKER_REPOS", str(tmp_path / "cache" / "repos"))
    with sandbox.pull_lock():
        pass
    assert (tmp_path / "cache" / "locks" / "pull.lock").exists()
    monkeypatch.delenv("UDOCKER_REPOS")
    monkeypatch.delenv("UDOCKER_LAYERS", raising=False)
    assert sandbox.pull_lock_directory().name.startswith("contextos-images-")


def test_image_complete_requires_every_manifest_blob(tmp_path, monkeypatch):
    monkeypatch.setenv("UDOCKER_DIR", str(tmp_path))
    monkeypatch.delenv("UDOCKER_REPOS", raising=False)
    monkeypatch.delenv("UDOCKER_LAYERS", raising=False)
    image = "ns/name:tag"
    directory = sandbox.image_directory(image)
    assert directory == tmp_path / "repos" / "ns" / "name" / "tag"
    assert sandbox.image_directory("plain") == tmp_path / "repos" / "plain" / "latest"
    assert not sandbox.image_complete(image)  # never pulled
    directory.mkdir(parents=True)
    (directory / "manifest").write_text('{"layers":[{"digest":"sha256:a"}],"config":{"digest":"sha256:c"}}')
    assert not sandbox.image_complete(image)  # pull in progress: no blob yet
    (directory / "sha256:a").write_text("x")
    assert not sandbox.image_complete(image)  # config blob still missing
    (tmp_path / "layers").mkdir()
    (tmp_path / "layers" / "sha256:c").write_text("x")
    assert sandbox.image_complete(image)
    (directory / "sha256:a").unlink()
    (directory / "sha256:a").symlink_to(tmp_path / "layers" / "sha256:gone")  # link into the store, blob lost
    assert not sandbox.image_complete(image)


def test_image_listed_while_another_job_pulls_it_is_pulled_not_used(tmp_path, monkeypatch):
    """`udocker images` lists an image from the moment its pull starts. Until every blob is in place the
    sandbox must pull it under the shared lock (which waits for the running pull) instead of creating a
    container from it, which fails with "getting layers or json"."""
    stub = tmp_path / "udocker"
    stub.write_text(STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("STUB_LOG", str(tmp_path / "calls.log"))
    monkeypatch.setenv("UDOCKER_DIR", str(tmp_path / "store"))
    monkeypatch.delenv("UDOCKER_REPOS", raising=False)
    monkeypatch.delenv("UDOCKER_LAYERS", raising=False)
    cfg = demo_config(tmp_path, "late_lookahead")
    cfg.sandbox.kind, cfg.sandbox.executable = "udocker", str(stub)
    task = PublicTask("repo__x-1", "fix it", "repo/x", "a" * 40, "xingyaoww/sweb.eval.x86_64.repo_s_x-1:latest")
    directory = sandbox.image_directory(task.image)
    directory.mkdir(parents=True)
    (directory / "manifest").write_text('{"layers":[{"digest":"sha256:l1"},{"digest":"sha256:l2"}],"config":{}}')
    (directory / "sha256:l1").write_text("blob")  # l2 is still downloading in another job
    (tmp_path / "calls.log.pulled").write_text(task.image + "\n")  # so the listing already shows it
    box = Udocker(cfg, task, "repo__x-1-0-late_lookahead")
    assert not box._have_image()
    assert box.ensure_image() is True  # pulled; the stub's pull completes the blobs
    assert box._have_image() and box.ensure_image() is False
    calls = [c.split()[1] for c in (tmp_path / "calls.log").read_text().splitlines()]
    assert calls.count("pull") == 1
