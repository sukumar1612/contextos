"""Numerical checks for the small controller's synchronous and decoupled PPO objectives."""

import copy
import math

import pytest
import torch

from contextos.config import Config
from contextos.policy import Controller


def controller(*, asynchronous=True, choice_only=True):
    cfg = Config()
    cfg.strategy.name = "ppo"
    cfg.encoder.semantic, cfg.encoder.dimension = False, 1
    cfg.learner.hidden = []  # isolate actor and critic derivatives
    cfg.learner.async_collection = asynchronous
    cfg.learner.choice_only = choice_only
    cfg.learner.epochs = 1
    cfg.learner.entropy = cfg.learner.value_weight = 0.0
    result = Controller(cfg)
    with torch.no_grad():
        for parameter in result.net.parameters():
            parameter.zero_()
    return result


def probabilities(policy, first, *, illegal_logit=0.0):
    with torch.no_grad():
        policy.net.actions.bias.copy_(torch.tensor([math.log(first), math.log1p(-first), illegal_logit]))


def row(policy, *, reward=1.0, version=None, forced=False, behavior=None, observation=0.0, terminal=True):
    obs = [0.0] * policy.observation_size
    obs[0] = observation
    mask = [True, not forced, False]
    if behavior is None:
        with torch.no_grad():
            logits, _ = policy.net(torch.tensor([obs]))
            behavior = 1.0 if forced else float(logits[0, :2].softmax(0)[0])
    return {
        "observation": obs,
        "next_observation": [0.0] * policy.observation_size,
        "mask": mask,
        "next_mask": [True, False, False],
        "action": 0,
        "log_prob": math.log(behavior),
        "value": 999.0,  # a stale value estimate must not enter the refreshed GAE calculation
        "version": policy.version if version is None else version,
        "reward": reward,
        "terminal": terminal,
    }


@pytest.mark.parametrize("asynchronous", [False, True])
def test_no_choice_or_empty_ppo_update_does_not_age_the_policy(asynchronous):
    policy = controller(asynchronous=asynchronous)
    policy.version = 5
    forced = row(policy, forced=True, version=0 if asynchronous else 5)
    original = copy.deepcopy(policy.net.state_dict())
    for episodes in ([{"name": "forced", "transitions": [forced]}], []):
        metrics = policy.update(episodes)
        assert metrics["version"] == policy.version == 5
        assert metrics["optimizer_steps"] == metrics["used_episodes"] == metrics["used_choices"] == 0
        assert metrics["used_episode_names"] == []
        assert all(torch.equal(value, original[key]) for key, value in policy.net.state_dict().items())


def test_stale_batch_is_accounted_for_without_aging_other_episodes():
    policy = controller()
    policy.version = 5
    episode = {"name": "stale", "transitions": [row(policy, version=1)]}
    metrics = policy.update([episode])
    assert metrics["received_episodes"] == metrics["dropped_stale_episodes"] == 1
    assert metrics["dropped_stale_episode_names"] == ["stale"]
    assert metrics["dropped_stale_choices"] == 1
    assert metrics["max_staleness"] == metrics["mean_staleness"] == 4
    assert metrics["version"] == policy.version == 5
    assert metrics["episodes"] == metrics["optimizer_steps"] == metrics["used_episodes"] == 0
    assert metrics["used_episode_names"] == []


@pytest.mark.parametrize("choice_only", [False, True])
def test_old_forced_prefix_does_not_reject_recent_choices(choice_only):
    policy = controller(choice_only=choice_only)
    policy.cfg.staleness_threshold, policy.version = 1, 5
    rows = [row(policy, forced=True, version=0, terminal=False), row(policy, version=4)]
    metrics = policy.update([{"name": "recent-choice", "transitions": rows}])
    assert metrics["dropped_stale_episodes"] == 0
    assert metrics["max_staleness"] == metrics["max_used_staleness"] == 1
    assert metrics["optimizer_steps"] == metrics["used_episodes"] == metrics["used_choices"] == 1
    assert metrics["used_episode_names"] == ["recent-choice"]
    assert metrics["training_transitions"] == metrics["used_transitions"] == (1 if choice_only else 2)


def test_a_recent_final_action_does_not_hide_an_old_genuine_choice():
    policy = controller()
    policy.cfg.staleness_threshold, policy.version = 1, 5
    rows = [row(policy, version=1, terminal=False), row(policy, version=5)]
    metrics = policy.update([{"transitions": rows}])
    assert metrics["max_staleness"] == 4
    assert metrics["mean_staleness"] == 2
    assert metrics["dropped_stale_choices"] == 2
    assert metrics["optimizer_steps"] == 0


def test_future_behavior_version_is_rejected():
    policy = controller()
    with pytest.raises(ValueError, match="future policy version"):
        policy.update([{"transitions": [row(policy, version=1)]}])


def test_ddqn_warmup_keeps_its_ingestion_version_for_replay_recovery():
    cfg = Config()
    cfg.strategy.name = "double_dqn"
    cfg.learner.choice_only = True
    policy = Controller(cfg)
    metrics = policy.update([{"transitions": []}])
    assert metrics["optimizer_steps"] == 0
    assert metrics["version"] == 1


@pytest.mark.parametrize("behavior, expected_weight", [(0.5, 1.5), (0.25, 2.0)])
def test_behavior_importance_weight_is_capped_and_has_no_gradient(monkeypatch, behavior, expected_weight):
    policy = controller()
    probabilities(policy, 0.75)
    gradients = []

    def capture(loss):
        policy.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients.append(policy.net.actions.bias.grad.detach().clone())
        policy.steps += 1

    monkeypatch.setattr(policy, "_step", capture)
    metrics = policy.update([{"transitions": [row(policy, reward=2.0, behavior=behavior)]}])
    # d[-w * 2 * pi(a)/pi_prox(a)] / d logits at pi=pi_prox=(.75,.25).
    torch.testing.assert_close(gradients[0], torch.tensor([-0.5, 0.5, 0.0]) * expected_weight)
    assert metrics["behavior_weight_mean"] == pytest.approx(expected_weight)
    assert metrics["behavior_weight_ess"] == metrics["behavior_weight_ess_fraction"] == 1
    assert metrics["behavior_weight_cap_fraction"] == (1 if behavior == 0.25 else 0)


def test_proximal_anchor_and_behavior_weight_stay_frozen_across_epochs(monkeypatch):
    policy = controller()
    policy.cfg.epochs, policy.cfg.behave_weight_threshold = 2, 10.0
    probabilities(policy, 0.5)
    losses, gradients = [], []

    def advance(loss):
        policy.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        losses.append(float(loss.detach()))
        gradients.append(float(policy.net.actions.bias.grad[0]))
        probabilities(policy, 0.55)
        policy.steps += 1

    monkeypatch.setattr(policy, "_step", advance)
    policy.update([{"transitions": [row(policy, reward=2.0, behavior=0.25)]}])
    assert losses == pytest.approx([-4.0, -4.4])
    assert gradients == pytest.approx([-2.0, -1.98])


def test_exact_masked_kl_stops_when_behavior_sampled_ratio_estimate_misses_drift(monkeypatch):
    policy = controller()
    policy.cfg.epochs, policy.cfg.kl_stop = 2, 0.03
    policy.version = 1
    probabilities(policy, 0.99, illegal_logit=100.0)

    def advance(loss):
        # A controlled first optimizer step moves the rare legal action from 1% to 10%.
        probabilities(policy, 0.90, illegal_logit=-100.0)
        policy.steps += 1

    monkeypatch.setattr(policy, "_step", advance)
    metrics = policy.update(
        [{"name": "sampled-common-action", "transitions": [row(policy, behavior=0.99999, version=0)]}]
    )
    ratio = 0.90 / 0.99
    old_sample_estimate = ratio - 1 - math.log(ratio)
    expected_kl = 0.99 * math.log(0.99 / 0.90) + 0.01 * math.log(0.01 / 0.10)
    assert old_sample_estimate < policy.cfg.kl_stop < expected_kl
    assert metrics["early_stop"] and metrics["optimizer_steps"] == 1
    assert metrics["max_kl"] == pytest.approx(expected_kl, abs=1e-6)
    assert metrics["approx_kl"] == metrics["exact_kl"] == pytest.approx(expected_kl / 2, abs=1e-6)
    assert metrics["used_episode_names"] == ["sampled-common-action"]


@pytest.mark.parametrize("asynchronous, second_loss, second_gradient", [(False, 5.0, 2.5), (True, 3.0, 0.0)])
def test_negative_advantage_dual_clip_is_specific_to_async_ppo(monkeypatch, asynchronous, second_loss, second_gradient):
    policy = controller(asynchronous=asynchronous)
    policy.cfg.epochs, policy.cfg.kl_stop = 2, 10.0
    probabilities(policy, 0.1)
    losses, gradients = [], []

    def advance(loss):
        policy.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        losses.append(float(loss.detach()))
        gradients.append(float(policy.net.actions.bias.grad[0]))
        probabilities(policy, 0.5)
        policy.steps += 1

    monkeypatch.setattr(policy, "_step", advance)
    metrics = policy.update([{"transitions": [row(policy, reward=-1.0, behavior=0.1)]}])
    assert losses == pytest.approx([1.0, second_loss])
    assert gradients[1] == pytest.approx(second_gradient)
    assert metrics["clip_fraction"] == 0.5


def test_weight_ess_uses_one_weight_per_genuine_choice():
    policy = controller()
    probabilities(policy, 0.25)
    episodes = [
        {"name": "lower-weight", "transitions": [row(policy, behavior=0.5, reward=1.0)]},
        {"name": "capped-weight", "transitions": [row(policy, behavior=0.05, reward=2.0)]},
    ]
    metrics = policy.update(episodes)
    expected_ess = (0.5 + 2.0) ** 2 / (0.5**2 + 2.0**2)
    assert metrics["behavior_weight_mean"] == pytest.approx(1.25)
    assert metrics["behavior_weight_max"] == pytest.approx(2.0)
    assert metrics["behavior_weight_cap_fraction"] == 0.5
    assert metrics["behavior_weight_ess"] == pytest.approx(expected_ess)
    assert metrics["behavior_weight_ess_fraction"] == pytest.approx(expected_ess / 2)
    assert metrics["used_episodes"] == metrics["used_choices"] == 2


def test_used_names_report_only_episodes_reached_before_kl_stop(monkeypatch):
    policy = controller()
    policy.cfg.batch_size = 1
    probabilities(policy, 0.99)
    episodes = [
        {"name": "stepped", "transitions": [row(policy, reward=1.0)]},
        {"name": "not-stepped", "transitions": [row(policy, reward=2.0)]},
    ]

    def advance(loss):
        probabilities(policy, 0.90)
        policy.steps += 1

    monkeypatch.setattr(policy, "_step", advance)
    monkeypatch.setattr(torch, "randperm", torch.arange)
    metrics = policy.update(episodes)
    assert metrics["received_episodes"] == metrics["episodes"] == 2
    assert metrics["early_stop"] and metrics["used_episodes"] == metrics["used_choices"] == 1
    assert metrics["used_episode_names"] == ["stepped"]


def test_gae_refreshes_values_and_discounts_collapsed_interactions(monkeypatch):
    policy = controller()
    policy.cfg.gamma, policy.cfg.gae_lambda, policy.cfg.value_weight = 0.5, 0.75, 1.0
    with torch.no_grad():
        policy.net.value.weight[0, 0] = 1.0
    rows = [
        row(policy, observation=1.0, reward=1.0, terminal=False),
        row(policy, observation=2.0, reward=2.0, forced=True, terminal=False),
        row(policy, observation=3.0, reward=4.0),
    ]
    gradients = []

    def capture(loss):
        policy.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients.append((float(policy.net.value.bias.grad[0]), float(policy.net.value.weight.grad[0, 0])))
        policy.steps += 1

    monkeypatch.setattr(policy, "_step", capture)
    policy.update([{"transitions": rows}])
    # R0=1+.5*2=2, duration=2, V=(1,3). A1=4-3=1;
    # A0=2+.5**2*3-1 + .5**2*.75*A1=1.9375, so returns=(2.9375,4).
    assert gradients[0] == pytest.approx((-2.9375, -4.9375))


@pytest.mark.parametrize(
    "name,value",
    [
        ("staleness_threshold", -1),
        ("staleness_threshold", 1.5),
        ("staleness_threshold", True),
        ("behave_weight_threshold", 0.5),
        ("behave_weight_threshold", float("inf")),
        ("behave_weight_threshold", float("nan")),
        ("clip_ratio_c", 1.0),
        ("clip_ratio_c", -1.0),
        ("clip_ratio_c", float("nan")),
    ],
)
def test_invalid_async_stability_settings_fail_before_collection(name, value):
    cfg = Config()
    cfg.strategy.name = "ppo"
    setattr(cfg.learner, name, value)
    with pytest.raises(ValueError, match=name):
        Controller(cfg)
