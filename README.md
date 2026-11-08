# ContextOS core

Context scheduling for a fixed coding agent: the harness from the ContextOS experiment reduced to its
mechanism, running against a self-hosted Qwen3.6-35B-A3B vLLM server.

The v5 correctness review, optional training profiles, reproducible offline smoke check and diagnostic
reader are in [`audit/README.md`](audit/README.md). See that report before transferring synchronous PPO
settings to async collection.

The follow-up [concept alignment review](audit/CONCEPT_REVIEW.md) checks the runtime against the original
ContextOS design, documents the remaining persistence simplifications, and records the additional
unusable-summary validation fix. The [text-feature update](audit/TEXT_FEATURES.md) enables a frozen Qwen
encoder at 256 dimensions per block, adds caching, and documents setup and checkpoint compatibility.

A fixed coding agent works on a SWE-Gym Lite issue inside a container with two tools, `bash` and `submit`.
Before every model request a controller looks at the conversation and picks one of three actions:

| Action | Effect |
|---|---|
| `CONTINUE` | send the prompt as it is |
| `PREPARE` | snapshot the old prefix of the conversation and summarize it on a background thread |
| `COMMIT` | replace that prefix with the finished summary |

The controller is a two-threshold rule (PREPARE at a soft level of utilization, COMMIT at a hard one) in
two presets, or a small MLP trained with PPO or Double DQN:

| Strategy | Soft (PREPARE) | Hard (COMMIT) |
|---|---|---|
| `smoothagent_lookahead` | 12,000 tokens | 16,000 tokens, SmoothAgent's defaults, as fractions of the 32,768 cap |
| `late_lookahead` | 85% of the cap | 95% of the cap |

Both presets can be overridden with `strategy.prepare_fraction` and `strategy.commit_fraction`;
`sync_at_limit` always continues until the shared guard performs synchronous compaction. The reward is
"issue resolved" minus the reference dollar cost of the tokens, with a small penalty whenever the harness has
to force a compaction because the policy let the prompt overflow. Wall-clock time is logged separately.

## Layout

```
src/contextos/
  types.py        Action, Decision, Reply, ToolResult, PublicTask, Grade, the two exceptions
  config.py       dataclass settings; YAML overlays and --set section.key=value
  tokens.py       Counter: HF chat-template counting or a byte estimate; truncate()
  history.py      pinned messages + interaction Groups; Snapshot; prefix / replacement / commit
  context.py      ContextRuntime: phase (IDLE/PREPARING/READY), mask, prepare, reconcile, commit, guard
  observe.py      12 scalars + two cached frozen embeddings (default 256-d each; 524 inputs)
  agent.py        the two tools, the system and summarizer prompts, parse_actions, error_messages
  provider.py     OpenAI-compatible client for one server; per-episode request accounting; FakeProvider
  policy.py       Rule presets; Network; Controller with PPO (sync or async) and Double DQN; make_strategy
  sandbox.py      Docker and udocker container lifecycle, git sanitize, exec, patch export; FakeSandbox
  grader.py       fresh-container grading through the pinned SWE-Gym fork
  grader_bridge.py  runs inside the grader's own venv: eval script, log parsing
  data.py         load rows, seeded split, public_task
  runner.py       run_episode(): the interaction loop; finalize(): reward attribution
  collector.py    the parallel, resumable collector: progress.json, retries, cohorts, checkpoints
  experiments.py  schedule, run directory and fingerprint, resume, prefetch, report()
  cli.py          demo | train | evaluate | report | prefetch | probe
```

Dependency direction is downward: `cli → experiments → collector → runner → {context, observe, policy,
agent, sandbox, grader} → {history, tokens, provider, data} → {config, types}`.

## Run it

```bash
uv sync --group dev                  # torch, numpy, httpx, pyyaml + pytest, ruff, mypy
uv run pytest -q                     # offline: fake provider, fake sandbox, zero embeddings
uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy
uv run contextos demo --strategy late_lookahead --episodes 1
uv run contextos demo --strategy ppo --episodes 4
```

For a real run the harness needs a vLLM server and the pinned grader:

```bash
uv sync --group dev --extra live     # transformers, sentence-transformers, datasets
uv run --extra live python audit/encoder_smoke.py --config configs/cluster.yaml
export LOCAL_QWEN_API_KEY=...        # or provider.api_key_file (default ~/vllm/api_key)
uv run contextos probe -c configs/cluster.yaml --set provider.base_url=http://host:port/v1
uv run contextos evaluate -c configs/cluster.yaml --set provider.base_url=http://host:port/v1 \
    --run runs/eval --partition test --workers 12
uv run contextos train -c configs/cluster.yaml --set provider.base_url=http://host:port/v1 \
    --set strategy.name=ppo --set learner.async_collection=true --run runs/ppo --workers 24
uv run contextos report --run runs/eval
```

On the SoC cluster everything runs as Slurm jobs; `cluster/README.md` documents the server script, the
campaign driver that keeps a 3-hour server alive across restarts, prefetching, and the analysis tools.

The dataset is SWE-Gym Lite, 230 instances: a seeded shuffle, then 170 train, 30 validation and 30 test
tasks (`benchmark.counts`, or `null` to split by `benchmark.ratios`). Rows are read from `benchmark.rows` and
downloaded on first use if missing (needs the `live` extra). The tokenizer must match the coding model
(`context.tokenizer`, a local directory on the cluster); with `null` the counter uses UTF-8 byte counts.

The controller uses a separate, frozen `Qwen/Qwen3-Embedding-0.6B` on the harness host. Both the default
configuration and `configs/cluster.yaml` enable it. Its 256-dimensional normalized work and ready-summary
vectors join the 12 scalar features. `encoder.max_tokens=2048` limits each input block; it is independent
of `encoder.dimension=256`. Run the encoder preflight once with network access before offline jobs.
Set `encoder.device=cuda` only if the harness process has a local GPU with room for the encoder; the remote
vLLM URL does not provide that device. Use a new run directory for text-enabled training; old numeric-only
checkpoints are not compatible. The optional `audit/configs/numeric_only.yaml` is a separate ablation.

## Provider

One HTTP client, one bearer token (`provider.api_key_env`, else `provider.api_key_file`), one concurrency
limit shared by coding and summary requests (`provider.max_concurrent_requests`). A 429 is retried, a 5xx is
retried and charged its worst-case reference cost (the server may have run it), any other error ends the
request. When the server stops answering, requests re-read `provider.endpoint_file` and wait up to
`provider.server_wait_seconds` for it or a replacement to answer, so a vLLM job hitting its time limit stalls
the episodes in flight instead of failing them. Costs are reference prices (`input_price`, `output_price`);
nothing is billed.

## Parallel collection and learning

`experiment.workers` (or `--workers`) runs that many episodes at once, each in its own container. Rules,
evaluation and Double DQN keep every slot busy and learn as episodes land. PPO runs fork-join cohorts of
`learner.episodes_per_update` episodes under one frozen policy, or, with `learner.async_collection`, keeps
every worker busy and learns from each cohort as it completes: a decoupled objective clips around the
proximal policy, weights stale samples by a capped proximal/behaviour ratio, and refreshes values with the
current critic before GAE. It drops an episode when its oldest genuine choice is more than
`learner.staleness_threshold` optimizer-bearing policy updates old. A nonzero `choices_per_update` replaces
the episode-count trigger in both collection modes. Workers read the shared controller at every decision,
so one async episode can contain several policy versions. Synchronous collection uses ordinary clipped
PPO; the async path also uses dual clipping. The async correction is approximate: ordinary GAE does not
fully correct stale future actions or the state distribution.

PPO logs received, retained, actually used and stale-dropped samples separately, plus importance-weight
concentration and exact legal-action KL stopping. See `audit/inspect_training.py` for a compact view.

`experiment.max_episode_seconds` caps an episode's agent loop (`TIME_LIMIT`, still graded). The image of a
task is removed after its last attempt (`sandbox.evict_images`); with udocker each image is extracted once
per node into a template that episodes and grades clone (`sandbox.template_containers`).

## Failures and resumption

- A failed episode is retried once, then skipped and recorded in `progress.json`. Three skipped episodes
  in a row abort the run.
- A failed background summary loses only that candidate; the episode continues and the guard still
  protects the cap.
- `progress.json` holds the launch cursor, the indices of episodes in flight, the names of finalized
  episodes not yet learned from, the update history and the failures. `policy.pt` holds the learner state;
  Double DQN's replay buffer is rebuilt from the episode files on resume.
- The checkpoint's `consumed_names` ledger is authoritative; `learned_names` is retained as a legacy alias.
  Consumption includes PPO stale drops and DDQN replay ingestion, so it does not mean every episode
  supplied an optimizer sample. Resume recovers published episodes even if progress was not yet written.
- Running the same `train` or `evaluate` command on the same directory continues from the cursor; episodes
  that were in flight run again, and their containers are reclaimed by name.

## How one interaction works

1. `runtime.reconcile()`: if a background summary is done and `delay_interactions` boundaries have passed
   since `PREPARE`, the job becomes `READY`. The delay makes the controller's experience independent of
   API latency.
2. `observe()` builds the observation vector; `runtime.mask()` says which of PREPARE / COMMIT are legal
   (never both: there is one job at a time).
3. `strategy.decide()` returns an action; a transition record is appended.
4. `runtime.apply()` performs PREPARE or COMMIT. `runtime.guard()` then checks the prompt fits; if not, it
   records a forced intervention and compacts.
5. `provider.complete()` sends the prompt; the cost is charged to this transition. Summary requests are
   charged to the transition that chose PREPARE.
6. Tool calls run in the container; output is truncated to a per-interaction token budget; the assistant
   message plus its tool results join the history as one group.

At the end, `finalize()` turns per-transition costs, forced interventions and the grade into rewards.
Waste is the cost of a valid summary never consumed by a later coding request, including a committed
summary that was not used; an invalid or failed summary costs tokens but is not waste, so a service failure
never looks like a bad decision. A tool command that hits the host-side timeout returns exit code 124.
