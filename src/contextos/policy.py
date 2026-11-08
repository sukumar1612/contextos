"""Threshold rules and the small masked controllers: PPO (synchronous or asynchronous) and Double DQN."""

from __future__ import annotations

import copy
import math
import os
import random
import threading
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, kl_divergence

from contextos.config import Config
from contextos.observe import OBSERVATION_SCHEMA_VERSION, SIZE, observation_size
from contextos.types import Action, Decision, validate_mask

PRESETS = {  # (prepare_fraction, commit_fraction) of the input cap
    "smoothagent_lookahead": (12_000 / 32_768, 16_000 / 32_768),  # SmoothAgent's soft 12,000 / hard 16,000 tokens
    "late_lookahead": (0.85, 0.95),
    "sync_at_limit": (float("inf"), float("inf")),
}
RULES = set(PRESETS)
LEARNED = {"ppo", "double_dqn"}


class Rule:
    """Two thresholds on utilization (obs[0]): PREPARE at the soft one, COMMIT at the hard one."""

    def __init__(self, cfg: Config):
        soft, hard = PRESETS[cfg.strategy.name]
        self.synchronous = cfg.strategy.name == "sync_at_limit"
        self.soft = soft if cfg.strategy.prepare_fraction is None else cfg.strategy.prepare_fraction
        self.hard = hard if cfg.strategy.commit_fraction is None else cfg.strategy.commit_fraction

    def decide(self, observation, mask, training=False) -> Decision:
        validate_mask(mask)
        if self.synchronous:
            return Decision(Action.CONTINUE)  # the common guard compacts when the cap is reached
        utilization = float(observation[0])
        if mask[2] and utilization >= self.hard:
            return Decision(Action.COMMIT)
        if mask[1] and utilization >= self.soft:
            return Decision(Action.PREPARE)
        return Decision(Action.CONTINUE)


def masked(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~masks.bool(), torch.finfo(logits.dtype).min)


def observation_identity(cfg: Config) -> dict:
    """Describe the input meaning, not just its width; device/cache do not change the features."""
    encoder = cfg.encoder
    identity: dict = {"scalar_schema": 1, "semantic": encoder.semantic, "dimension": encoder.dimension}
    if encoder.semantic:
        identity.update(
            text_schema=OBSERVATION_SCHEMA_VERSION,
            model=encoder.model,
            revision=encoder.revision,
            max_tokens=encoder.max_tokens,
        )
    return identity


class Network(nn.Module):
    def __init__(self, hidden: list[int], critic: bool, input_size: int = SIZE):
        super().__init__()
        layers, width = [], input_size
        for size in hidden:
            layers += [nn.Linear(width, size), nn.Tanh()]
            width = size
        self.body = nn.Sequential(*layers)
        self.actions = nn.Linear(width, 3)
        self.value = nn.Linear(width, 1) if critic else None

    def forward(self, x):
        h = self.body(x)
        return self.actions(h), (self.value(h).squeeze(-1) if self.value is not None else None)


def gae(episode: list[dict], gamma: float, lam: float):
    advantages, carry = np.zeros(len(episode), np.float32), 0.0
    for i in reversed(range(len(episode))):
        t = episode[i]
        next_value = 0.0 if t["terminal"] else episode[i + 1]["value"]
        discount = gamma ** t.get("duration", 1)
        delta = t["reward"] + discount * next_value - t["value"]
        carry = delta + discount * lam * (not t["terminal"]) * carry
        advantages[i] = carry
    return advantages, advantages + np.array([t["value"] for t in episode], np.float32)


def choice_transitions(transitions: list[dict], gamma: float) -> list[dict]:
    """Join each genuine choice to the next one, carrying intervening costs and the outcome.

    The forced prefix before the first choice is independent of controller actions. It
    remains in episode cost/return accounting, but has no decision transition. GAE's
    lambda advances per decision; gamma still discounts elapsed interactions.
    """
    indices = [i for i, t in enumerate(transitions) if sum(t["mask"]) > 1]
    settled = []
    for start, stop in zip(indices, indices[1:] + [len(transitions)], strict=False):
        row, last = dict(transitions[start]), transitions[stop - 1]
        row.update(
            reward=sum(gamma**j * t["reward"] for j, t in enumerate(transitions[start:stop])),
            duration=stop - start,
            next_observation=last["next_observation"],
            next_mask=last["next_mask"],
            terminal=last["terminal"],
        )
        settled.append(row)
    return settled


class Controller:
    def __init__(self, cfg: Config):
        self.cfg, self.algorithm = cfg.learner, cfg.strategy.name
        if self.algorithm not in LEARNED:
            raise ValueError(f"Not a learned controller: {self.algorithm}")
        for name in (
            "batch_size",
            "epochs",
            "episodes_per_update",
            "target_interval",
            "transitions_per_update",
            "epsilon_choices",
            "replay_capacity",
        ):
            if not isinstance(getattr(self.cfg, name), int) or getattr(self.cfg, name) <= 0:
                raise ValueError(f"learner.{name} must be a positive integer")
        if self.cfg.choices_per_update < 0 or self.cfg.learning_starts < 0:
            raise ValueError("choices_per_update and learning_starts must be nonnegative")
        if not 0 <= self.cfg.gamma <= 1 or not 0 <= self.cfg.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must lie between zero and one")
        if self.cfg.target_clock not in {"updates", "choices"}:
            raise ValueError("learner.target_clock must be 'updates' or 'choices'")
        if self.cfg.target_clock == "choices" and not self.cfg.choice_only:
            raise ValueError("target_clock='choices' requires choice_only=true")
        if self.algorithm == "ppo":
            if (
                not isinstance(self.cfg.staleness_threshold, int)
                or isinstance(self.cfg.staleness_threshold, bool)
                or self.cfg.staleness_threshold < 0
            ):
                raise ValueError("learner.staleness_threshold must be a nonnegative integer")
            for name, minimum, inclusive in (
                ("behave_weight_threshold", 1.0, True),
                ("clip_ratio_c", 1.0, False),
            ):
                value = getattr(self.cfg, name)
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    or (value < minimum if inclusive else value <= minimum)
                ):
                    comparison = "at least" if inclusive else "greater than"
                    raise ValueError(f"learner.{name} must be finite and {comparison} {minimum}")
        torch.manual_seed(cfg.experiment.seed)
        self.random = random.Random(cfg.experiment.seed)
        self.device = torch.device(self.cfg.device)
        self.observation_size = observation_size(cfg)
        self.observation_identity = observation_identity(cfg)
        self.net = Network(self.cfg.hidden, self.algorithm == "ppo", self.observation_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.cfg.learning_rate)
        self.target = copy.deepcopy(self.net).eval() if self.algorithm == "double_dqn" else None
        self.replay: deque[dict] = deque(maxlen=self.cfg.replay_capacity)
        self.version = self.choice_points = self.steps = self.inserted = 0
        self.checkpoint_metadata: dict = {}
        self.lock = threading.RLock()  # episodes decide on worker threads while the main thread updates

    def decide(self, observation, mask, training=True) -> Decision:
        with self.lock:
            return self._decide(observation, mask, training)

    def _decide(self, observation, mask, training) -> Decision:
        validate_mask(mask)
        obs = torch.as_tensor(np.asarray(observation), dtype=torch.float32, device=self.device)[None]
        legal = torch.tensor(mask, device=self.device)[None]
        with torch.no_grad():
            logits, value = self.net(obs)
            logits = masked(logits, legal)
            if self.algorithm == "double_dqn":
                fraction = min(1.0, self.choice_points / self.cfg.epsilon_choices)
                epsilon = self.cfg.epsilon_start + fraction * (self.cfg.epsilon_end - self.cfg.epsilon_start)
                explore = training and self.random.random() < epsilon
                action = self.random.choice([i for i, ok in enumerate(mask) if ok]) if explore else int(logits.argmax())
                log_prob = None
            else:
                dist = Categorical(logits=logits)
                action = int(dist.sample()) if training else int(logits.argmax())
                log_prob = float(dist.log_prob(torch.tensor([action], device=self.device)))
        if training and sum(mask) > 1:
            self.choice_points += 1
        return Decision(Action(action), log_prob, float(value[0]) if value is not None else 0.0, self.version)

    def _t(self, value, dtype=torch.float32):
        return torch.as_tensor(np.asarray(value), dtype=dtype, device=self.device)

    def _step(self, loss):
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()
        self.steps += 1

    def update(self, episodes: list[dict]) -> dict:
        with self.lock:
            received = len(episodes)
            versions = {t["version"] for e in episodes for t in e["transitions"]}
            stale = {}
            if self.algorithm == "ppo" and self.cfg.async_collection:
                # Forced actions have probability one under every policy. In particular an old forced
                # prefix must not make recent genuine choices stale, in either time representation.
                ages = [[self.version - t["version"] for t in e["transitions"] if sum(t["mask"]) > 1] for e in episodes]
                if any(age < 0 for episode_ages in ages for age in episode_ages):
                    raise ValueError("Asynchronous batch contains a future policy version")
                kept: list[dict] = []
                dropped: list[dict] = []
                for episode, episode_ages in zip(episodes, ages, strict=True):
                    (kept if max(episode_ages, default=0) <= self.cfg.staleness_threshold else dropped).append(episode)
                choice_ages = [age for episode_ages in ages for age in episode_ages]
                stale = {
                    "dropped_stale_episodes": len(dropped),
                    "dropped_stale_episode_names": [e["name"] for e in dropped if "name" in e],
                    "dropped_stale_choices": sum(sum(t["mask"]) > 1 for e in dropped for t in e["transitions"]),
                    "max_staleness": max(choice_ages, default=0),
                    "mean_staleness": float(np.mean(choice_ages)) if choice_ages else 0.0,
                }
                episodes = kept
            elif self.algorithm != "double_dqn" and versions and versions != {self.version}:
                raise ValueError("On-policy batch contains another policy version")
            before = self.steps
            metrics = {**stale, **{"ppo": self._ppo, "double_dqn": self._dqn}[self.algorithm](episodes)}
            # PPO versions measure policy updates, not rejected/empty cohorts. DDQN versions retain
            # their ingestion-clock meaning for checkpoint/replay reconstruction, including warmup.
            if self.algorithm == "double_dqn" or self.steps > before:
                self.version += 1
            return {
                **metrics,
                "version": self.version,
                "optimizer_steps": self.steps - before,
                "total_optimizer_steps": self.steps,
                "received_episodes": received,
                "episodes": len(episodes),
                "raw_transitions": sum(len(e["transitions"]) for e in episodes),
                "choices": sum(sum(t["mask"]) > 1 for e in episodes for t in e["transitions"]),
                "choice_points": self.choice_points,
            }

    def training_transitions(self, episode: dict) -> list[dict]:
        return (
            choice_transitions(episode["transitions"], self.cfg.gamma)
            if self.cfg.choice_only
            else episode["transitions"]
        )

    def _ppo(self, episodes):
        trajectories = [self.training_transitions(e) for e in episodes]
        rows = [t for trajectory in trajectories for t in trajectory]
        empty_metrics = {
            "loss": 0.0,
            "approx_kl": 0.0,  # compatibility alias; this is now the exact categorical KL
            "exact_kl": 0.0,
            "max_kl": 0.0,
            "clip_fraction": 0.0,
            "early_stop": False,
            "training_transitions": len(rows),
            "used_episodes": 0,
            "used_episode_names": [],
            "used_transitions": 0,
            "used_choices": 0,
            "max_used_staleness": 0,
            "mean_used_staleness": 0.0,
            "behavior_weight_mean": 0.0,
            "behavior_weight_max": 0.0,
            "behavior_weight_cap_fraction": 0.0,
            "behavior_weight_ess": 0.0,
            "behavior_weight_ess_fraction": 0.0,
        }
        if not rows:
            return empty_metrics
        obs, masks = self._t([t["observation"] for t in rows]), self._t([t["mask"] for t in rows], torch.bool)
        actions, behaviour = self._t([t["action"] for t in rows], torch.long), self._t([t["log_prob"] for t in rows])
        choice = masks.sum(1) > 1  # only real decisions carry a policy gradient
        # Decoupled PPO: clip around the frozen policy at the start of this update, with a capped
        # proximal/behaviour weight for asynchronous samples. This is an approximate off-policy method:
        # refreshing the critic does not correct stale future actions in ordinary GAE (it is not V-trace).
        # Synchronous collection uses standard PPO, without importance weighting or async dual clipping.
        with torch.no_grad():
            logits, fresh = self.net(obs)
            proximal_dist = Categorical(logits=masked(logits, masks))
            proximal_logits = proximal_dist.logits
            proximal = proximal_dist.log_prob(actions)
            log_weight = proximal - behaviour
            cap = math.log(self.cfg.behave_weight_threshold)
            # Clamp before exponentiating so a very small behaviour probability cannot overflow.
            weight = log_weight.clamp(max=cap).exp() if self.cfg.async_collection else torch.ones_like(proximal)
            if choice.any():
                weights = weight[choice]
                ess = weights.sum().square() / weights.square().sum().clamp_min(torch.finfo(weights.dtype).tiny)
                empty_metrics.update(
                    behavior_weight_mean=float(weights.mean()),
                    behavior_weight_max=float(weights.max()),
                    behavior_weight_cap_fraction=(
                        float((log_weight[choice] > cap).float().mean()) if self.cfg.async_collection else 0.0
                    ),
                    behavior_weight_ess=float(ess),
                    behavior_weight_ess_fraction=float(ess / len(weights)),
                )
        for t, value in zip(rows, fresh.tolist(), strict=False):
            t["value"] = float(value)
        pairs = [gae(trajectory, self.cfg.gamma, self.cfg.gae_lambda) for trajectory in trajectories if trajectory]
        adv, ret = self._t(np.concatenate([a for a, _ in pairs])), self._t(np.concatenate([r for _, r in pairs]))
        if choice.sum() > 1:
            adv[choice] = (adv[choice] - adv[choice].mean()) / (adv[choice].std(unbiased=False) + 1e-8)
        losses, kls, early = [], [], False
        used_rows: set[int] = set()
        clipped_count = checked_choices = 0
        for _ in range(self.cfg.epochs):
            order = torch.randperm(len(rows)).tolist()
            for start in range(0, len(order), self.cfg.batch_size):
                idx = order[start : start + self.cfg.batch_size]
                logits, values = self.net(obs[idx])
                dist = Categorical(logits=masked(logits, masks[idx]))
                delta = dist.log_prob(actions[idx]) - proximal[idx]
                ratio, chosen = delta.exp(), choice[idx]
                policy_loss = entropy = values.sum() * 0
                if chosen.any():
                    clipped = ratio.clamp(1 - self.cfg.clip, 1 + self.cfg.clip)
                    surrogate = torch.minimum(ratio * adv[idx], clipped * adv[idx])
                    if self.cfg.async_collection:
                        surrogate = torch.where(
                            adv[idx] < 0, torch.maximum(surrogate, self.cfg.clip_ratio_c * adv[idx]), surrogate
                        )
                    policy_loss = -(weight[idx] * surrogate)[chosen].mean()
                    entropy = dist.entropy()[chosen].mean()
                    # Behaviour actions are not sampled from the proximal policy when collection is
                    # asynchronous, so an unweighted sampled-ratio KL is biased. Three actions make
                    # exact KL(proximal || current) cheap; invalid actions contribute zero probability.
                    exact_kl = kl_divergence(Categorical(logits=proximal_logits[idx]), dist)
                    kls.append(float(exact_kl[chosen].mean().detach().clamp_min(0)))
                    clipped_count += int(((ratio - 1).abs() > self.cfg.clip)[chosen].sum())
                    checked_choices += int(chosen.sum())
                    if kls[-1] > self.cfg.kl_stop:
                        early = True
                        break
                loss = (
                    policy_loss
                    + self.cfg.value_weight * nn.functional.mse_loss(values, ret[idx])
                    - self.cfg.entropy * entropy
                )
                self._step(loss)
                used_rows.update(idx)
                losses.append(float(loss.detach()))
            if early:
                break
        row_episodes = [i for i, trajectory in enumerate(trajectories) for _ in trajectory]
        used_episodes = sorted({row_episodes[i] for i in used_rows})
        used_ages = [self.version - rows[i]["version"] for i in used_rows if sum(rows[i]["mask"]) > 1]
        return {
            **empty_metrics,
            "loss": float(np.mean(losses or [0])),
            "approx_kl": float(np.mean(kls or [0])),
            "exact_kl": float(np.mean(kls or [0])),
            "max_kl": max(kls, default=0.0),
            "clip_fraction": clipped_count / checked_choices if checked_choices else 0.0,
            "early_stop": early,
            "used_episodes": len(used_episodes),
            "used_episode_names": [episodes[i]["name"] for i in used_episodes if "name" in episodes[i]],
            "used_transitions": len(used_rows),
            "used_choices": len(used_ages),
            "max_used_staleness": max(used_ages, default=0),
            "mean_used_staleness": float(np.mean(used_ages)) if used_ages else 0.0,
        }

    @staticmethod
    def replay_item(t: dict) -> dict:
        """The replay form of a transition; used both when learning and when rebuilding from disk."""
        return {
            "observation": np.asarray(t["observation"], dtype=np.float32),
            "next_observation": np.asarray(t["next_observation"], dtype=np.float32),
            "mask": np.asarray(t["mask"], dtype=bool),
            "next_mask": np.asarray(t["next_mask"], dtype=bool),
            "action": int(t["action"]),
            "reward": float(t["reward"]),
            "terminal": bool(t["terminal"]),
            "duration": int(t.get("duration", 1)),
        }

    def _dqn(self, episodes):
        assert self.target is not None
        before = self.inserted
        for e in episodes:
            for t in self.training_transitions(e):
                self.replay.append(self.replay_item(t))
                self.inserted += 1

        def due(n):
            return max(0, n - self.cfg.learning_starts + 1) // self.cfg.transitions_per_update

        losses = []
        for _ in range(due(self.inserted) - due(before)):
            batch = self.random.sample(list(self.replay), min(self.cfg.batch_size, len(self.replay)))
            obs, nxt = self._t([b["observation"] for b in batch]), self._t([b["next_observation"] for b in batch])
            next_masks = self._t([b["next_mask"] for b in batch], torch.bool)
            rewards, terminals = (
                self._t([b["reward"] for b in batch]),
                self._t([b["terminal"] for b in batch], torch.bool),
            )
            discounts = self._t([self.cfg.gamma ** b["duration"] for b in batch])
            actions = self._t([b["action"] for b in batch], torch.long)
            with torch.no_grad():  # select with the online net, evaluate with the target
                selection = masked(self.net(nxt)[0], next_masks).argmax(-1, keepdim=True)
                continuation = self.target(nxt)[0].gather(1, selection).squeeze(1)
                expected = rewards + discounts * (~terminals).float() * continuation
            predicted = self.net(obs)[0].gather(1, actions[:, None]).squeeze(1)
            loss = nn.functional.smooth_l1_loss(predicted, expected)
            self._step(loss)
            losses.append(float(loss.detach()))
            if self.cfg.target_clock == "updates" and self.steps % self.cfg.target_interval == 0:
                self.target.load_state_dict(self.net.state_dict())
        # Collected episodes settle atomically, so a choice-clock copy occurs after
        # the first settled episode that crosses the configured interval.
        if (
            self.cfg.target_clock == "choices"
            and self.inserted // self.cfg.target_interval > before // self.cfg.target_interval
        ):
            self.target.load_state_dict(self.net.state_dict())
        return {
            "loss": float(np.mean(losses or [0])),
            "replay_size": len(self.replay),
            "updates": len(losses),
            "training_transitions": self.inserted - before,
            "settled_transitions": self.inserted,
        }

    def state(self) -> dict:
        """Everything needed to continue training exactly where it stopped, except the replay buffer,
        which experiments.rebuild_replay reconstructs from the episode files on disk."""
        return {
            "algorithm": self.algorithm,
            "observation_size": self.observation_size,
            "observation_identity": copy.deepcopy(self.observation_identity),
            "network": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "target": self.target.state_dict() if self.target is not None else None,
            "random": self.random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "version": self.version,
            "steps": self.steps,
            "choice_points": self.choice_points,
            "inserted": self.inserted,
            "metadata": copy.deepcopy(self.checkpoint_metadata),
        }

    def save(self, path: Path, metadata: dict | None = None) -> None:
        with self.lock:
            if metadata is not None:
                self.checkpoint_metadata = copy.deepcopy(metadata)
            tmp = path.with_name(path.name + ".tmp")
            torch.save(self.state(), tmp)
            os.replace(tmp, path)

    def load(self, path: Path, full: bool) -> None:
        """full=True restores training state; full=False takes only the weights, for evaluation."""
        saved = torch.load(path, map_location=self.device, weights_only=False)  # local, trusted file
        if saved["algorithm"] != self.algorithm:
            raise ValueError("Checkpoint algorithm mismatch")
        if saved.get("observation_size", SIZE) != self.observation_size:
            raise ValueError("Checkpoint observation size mismatch; use the original encoder.dimension")
        if "observation_identity" not in saved:
            if self.observation_identity["semantic"]:
                raise ValueError(
                    "Legacy checkpoint has no encoder identity and cannot be loaded with text features. "
                    "Start a new semantic run; legacy numeric weights require encoder.semantic=false "
                    "and the original encoder.dimension."
                )
        elif saved["observation_identity"] != self.observation_identity:
            raise ValueError(
                "Checkpoint encoder identity mismatch: semantic mode, model/revision, token budget, "
                "dimension and text preprocessing must match. Use a matching configuration or train a new controller."
            )
        # All input-identity checks precede loading weights, optimizer state, counters or RNG.
        self.net.load_state_dict(saved["network"])
        self.version = saved["version"]
        self.checkpoint_metadata = saved.get("metadata", {})
        if full and "optimizer" in saved:
            self.optimizer.load_state_dict(saved["optimizer"])
            if self.target is not None and saved.get("target") is not None:
                self.target.load_state_dict(saved["target"])
            self.random.setstate(saved["random"])
            torch.set_rng_state(saved["torch_rng"].cpu())
            self.steps, self.choice_points, self.inserted = saved["steps"], saved["choice_points"], saved["inserted"]
        elif self.target is not None:
            self.target.load_state_dict(self.net.state_dict())


def make_strategy(cfg: Config):
    if cfg.strategy.name in RULES:
        return Rule(cfg)
    if cfg.strategy.name not in LEARNED:
        raise ValueError(f"Unknown strategy {cfg.strategy.name}")
    controller = Controller(cfg)
    if cfg.strategy.checkpoint:
        controller.load(Path(cfg.strategy.checkpoint), full=False)
    return controller
