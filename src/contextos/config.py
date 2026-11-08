"""Plain dataclass configuration. YAML files overlay in order; unknown keys are errors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Context:
    input_tokens: int = 32768  # hard cap on the serialized prompt
    margin_tokens: int = 1024  # safety margin added to every guarded count
    recent_tokens: int = 8192  # youngest groups that are never summarized
    minimum_prefix_tokens: int = 8192  # PREPARE is legal only above this prefix size
    minimum_gain_tokens: int = 2048  # COMMIT is legal only if it saves this much
    tool_output_tokens: int = 8192  # per-interaction allowance for tool output
    delay_interactions: int = 2  # controlled readiness: summary visible after this many boundaries
    summary_output_tokens: int = 4096
    summary_rounds: int = 4  # chunked summarization rounds
    emergency_rebuilds: int = 2  # forced compactions tried before giving up
    tokenizer: str | None = None  # HF tokenizer ID or local dir; None = byte estimate
    mode: str = "controlled_delay"  # or "natural_async"


@dataclass
class Provider:
    kind: str = "http"  # or "fake"
    base_url: str = "http://localhost:8000/v1"  # the vLLM server (cluster jobs pass the live address)
    model: str = "qwen3.6-35b-a3b"  # served model name
    api_key_env: str = "LOCAL_QWEN_API_KEY"  # environment variable holding the bearer token
    api_key_file: str | None = "~/vllm/api_key"  # read when the variable is unset; never written anywhere
    max_concurrent_requests: int = 24  # coding and summaries share this process-wide limit
    timeout_seconds: float = 180
    retries: int = 5
    input_price: float = 0.157019  # USD per million tokens, reference prices; only the reward uses these
    output_price: float = 0.507577
    agent_temperature: float = 0.2
    output_tokens: int = 4096
    reasoning_effort: str | None = None  # "low" | "high" | "max"; sent on every request and used when counting
    chat_template_kwargs: dict | None = (
        None  # template switches sent with every request, e.g. {"enable_thinking": false}
    )
    endpoint_file: str | None = (
        None  # read while the server is unreachable: holds the current base URL (a replaced server)
    )
    server_wait_seconds: float = (
        3600  # a request waits this long for an unreachable server to answer again; 0 = fail at once
    )


@dataclass
class Encoder:
    semantic: bool = True
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    revision: str | None = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"  # pin weights, tokenizer and pooling
    device: str = "cpu"
    max_tokens: int = 2048  # independent input-token cap for each text block
    dimension: int = 256  # Qwen MRL prefix, renormalized; 12 + 2 * 256 = 524-value state
    cache_size: int = 1024  # bounded, shared in-memory text/vector cache; 0 disables caching


@dataclass
class Strategy:
    name: str = "smoothagent_lookahead"  # smoothagent_lookahead | late_lookahead | sync_at_limit | ppo | double_dqn
    prepare_fraction: float | None = None  # override the preset's soft threshold (fraction of the cap)
    commit_fraction: float | None = None  # override the preset's hard threshold
    checkpoint: str | None = None  # trained weights when evaluating a learned policy


@dataclass
class Reward:
    success: float = 10
    cost: float = 1  # per USD spent
    forced: float = 0.05  # per forced intervention
    waste: float = 0  # per USD of summaries that were never committed
    potential: float = 0  # optional utilization potential; terminal potential is zero


@dataclass
class Learner:
    hidden: list[int] = field(default_factory=lambda: [128, 128])
    learning_rate: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy: float = 0.01
    value_weight: float = 0.5
    epochs: int = 4
    batch_size: int = 64
    episodes_per_update: int = 16
    choices_per_update: int = 0  # PPO: optional settled-choice trigger; replaces the episode trigger
    choice_only: bool = False  # collapse forced-action intervals for PPO / DDQN
    max_grad_norm: float = 0.5
    kl_stop: float = 0.03
    replay_capacity: int = 50000
    learning_starts: int = 1024
    transitions_per_update: int = 4
    target_interval: int = 1000
    target_clock: str = "updates"  # "choices": settled replay insertions instead of optimizer steps
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_choices: int = 20000
    async_collection: bool = (
        False  # PPO: keep every worker busy and learn from each cohort as it completes (cluster runs enable it)
    )
    staleness_threshold: int = 2  # async PPO: drop if the oldest genuine choice is more policy updates behind
    behave_weight_threshold: float = 2.0  # async PPO: cap on the proximal/behaviour importance weight
    clip_ratio_c: float = 3.0  # async PPO: dual clip bound for negative advantages
    device: str = "cpu"


@dataclass
class Sandbox:
    kind: str = "docker"  # docker | udocker | fake
    executable: str | None = None  # the docker or udocker binary; None = look it up on PATH
    udocker_mode: str = "F3"  # fakechroot modes F3/F4 need no user namespaces or daemon
    pull_retries: int = 3  # image pulls retry with backoff; registries and NAT links flap
    udocker_env: dict = field(
        default_factory=lambda: {  # udocker has no network isolation; point
            k: "http://127.0.0.1:9" for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")
        }
    )  # egress at a dead port
    workdir: str = "/testbed"
    tool_timeout: int = 120
    grader_timeout: int = 1800
    memory: str = "8g"
    cpus: float = 2
    pids_limit: int = 256
    max_output_bytes: int | None = 1_000_000  # None preserves complete grader logs; agent tools stay capped
    activation: str = "source /opt/miniconda3/bin/activate && conda activate testbed"
    evict_images: bool = True  # remove a task's image after its last attempt
    template_containers: bool = True  # udocker: extract each image once, clone it per episode and per grade
    platform: str | None = "linux/amd64"  # the SWE-Gym images are x86_64; requested explicitly on every host
    namespace: str = ""  # supplied per run to prevent cross-run container collisions
    minimum_free_gb: float = 0  # low-water mark before starting/pulling an episode
    disk_path: str | None = None  # filesystem that contains the local Docker data root


@dataclass
class Benchmark:
    dataset: str = "SWE-Gym/SWE-Gym-Lite"
    rows: str = "data/swe_gym_lite.jsonl"
    counts: dict | None = field(default_factory=lambda: {"train": 170, "validation": 30, "test": 30})
    ratios: dict = field(
        default_factory=lambda: {"train": 0.75, "validation": 0.125, "test": 0.125}
    )  # used when counts is null
    seed: int = 42
    image_template: str = "xingyaoww/sweb.eval.x86_64.{instance_id}:latest"
    grader_python: str = "external/grader/.venv/bin/python"
    grader_root: str = "external/grader/SWE-Bench-Fork"


@dataclass
class Experiment:
    seed: int = 42
    max_interactions: int = 200
    attempts_per_task: int = 4
    train_tasks: int | None = None  # None = every training task
    evaluation_repeats: int = 2
    evaluation_tasks: int | None = None  # fixed validation prefix for time-limited screening; test stays complete
    exclude_tasks: list[str] = field(
        default_factory=list
    )  # task IDs removed from their partitions after the seeded split
    workers: int = 1  # concurrent episodes; 1 reproduces a sequential run exactly
    episode_retries: int = 1  # re-run a failed episode this many times before skipping it
    max_consecutive_failures: int = 3  # abort the run after this many skipped episodes in a row
    max_episode_seconds: float | None = (
        None  # wall-clock cap on the agent loop; the episode ends as TIME_LIMIT and is graded
    )


@dataclass
class Config:
    context: Context = field(default_factory=Context)
    provider: Provider = field(default_factory=Provider)
    encoder: Encoder = field(default_factory=Encoder)
    strategy: Strategy = field(default_factory=Strategy)
    reward: Reward = field(default_factory=Reward)
    learner: Learner = field(default_factory=Learner)
    sandbox: Sandbox = field(default_factory=Sandbox)
    benchmark: Benchmark = field(default_factory=Benchmark)
    experiment: Experiment = field(default_factory=Experiment)


def load_config(paths: Sequence[str | Path] = (), overrides: Sequence[str] = ()) -> Config:
    cfg = Config()
    layers = [yaml.safe_load(Path(p).read_text()) or {} for p in paths]
    for override in overrides:  # section.key=value
        key, _, value = override.partition("=")
        section, name = key.split(".", 1)
        layers.append({section: {name: yaml.safe_load(value)}})
    for layer in layers:
        for section, values in layer.items():
            if not hasattr(cfg, section) or not isinstance(values, dict):
                raise ValueError(f"Unknown or malformed section: {section}")
            target = getattr(cfg, section)
            for name, value in values.items():
                if not hasattr(target, name):
                    raise ValueError(f"Unknown setting {section}.{name}")
                setattr(target, name, value)
    return cfg
