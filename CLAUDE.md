# ContextOS harness

Context scheduling for a fixed coding agent on SWE-Gym Lite: before each model request a controller
(a two-threshold rule or a small MLP trained with PPO or Double DQN) chooses CONTINUE, PREPARE or
COMMIT for the conversation prefix. `README.md` explains the mechanism, the module layout and the
dependency direction between modules; read it before changing `src/`.

## What is versioned

The repository holds the harness only: `src/`, `tests/`, `configs/`, `data/swe_gym_lite.jsonl`,
`pyproject.toml`, `uv.lock`. Everything else that may sit beside it is local-only and ignored:

| Folder | Holds | Rule |
|---|---|---|
| `runs/` | run directories, episode files, checkpoints, logs | outputs, never versioned |
| `cluster/` | job scripts and analysis helpers for the compute cluster | deployment tooling |
| `audit/`, `contextos_next_round/` | external review material and experiment packages | reference, not product code |
| `external/`, `.venv/` | the pinned grader fork and the virtualenv | installed, not versioned |

A fresh clone therefore has no runs, no cluster tooling and no review material. Do not assume they exist.

## Working on the code

1. `uv sync --group dev` once; `--extra live` adds the encoder and dataset dependencies for real runs.
2. Change `src/` and the matching tests together. The suite runs offline against the fake provider and
   fake sandbox; keep it free of network, model and container calls.
3. Done means `uv run pytest -q`, `uv run ruff check src tests`, `uv run ruff format --check src tests`
   and `uv run mypy src` are all clean.

## Conventions worth knowing

- Configuration is `Config` dataclass defaults, then `-c` YAML overlays in order, then `--set section.key=value`.
  Unknown keys are rejected, so a stale override fails the run at startup rather than being ignored.
- A run directory is resumable state. Re-running the same command on the same directory continues it;
  a new experiment gets a new directory. Never edit `progress.json` or `policy.pt` by hand, and never
  reuse a checkpoint across a change to the observation layout.
- Grading must see the complete test output: the grader uncaps capture for itself and refuses truncated
  or interrupted logs. Keep agent tool output capped and grading output uncapped.
- The bearer token for the model server comes from `LOCAL_QWEN_API_KEY` or `provider.api_key_file`.
  Keys never go into files under version control or into printed output.
- Episode records are gzipped JSON under `<run>/episodes/`; `contextos report --run <dir>` summarizes a run.

## Working with the user

Offline tests and local analysis run freely. Anything that spends model-server time, cluster allocations
or money is started only after the user says to start it, and cancelled on request.
