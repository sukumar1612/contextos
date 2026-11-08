"""Independent numeric and continuation checks for the Double DQN learner."""

import copy
import gzip
import json

import numpy as np
import pytest
import torch

from contextos.collector import rebuild_replay
from contextos.config import Config
from contextos.policy import Controller
from contextos.types import Action


def settings():
    cfg = Config()
    cfg.strategy.name = "double_dqn"
    cfg.encoder.semantic, cfg.encoder.dimension = False, 2
    cfg.learner.hidden = []
    cfg.learner.learning_starts = 1
    cfg.learner.transitions_per_update = 1
    cfg.learner.batch_size = 1
    cfg.learner.target_interval = 10_000
    return cfg


def row(controller, *, reward=1.0, terminal=True, mask=None, next_mask=None, duration=1, mark=0.0):
    observation = np.zeros(controller.observation_size, np.float32)
    observation[0] = mark
    return {
        "observation": observation.tolist(),
        "next_observation": observation.tolist(),
        "mask": mask or [True, True, False],
        "next_mask": next_mask or [True, False, False],
        "action": 0,
        "log_prob": None,
        "value": 0.0,
        "version": controller.version,
        "reward": reward,
        "terminal": terminal,
        "duration": duration,
    }


def constant_values(controller, online, target):
    with torch.no_grad():
        controller.net.actions.weight.zero_()
        controller.net.actions.bias.copy_(torch.tensor(online, dtype=torch.float32))
        controller.target.actions.weight.zero_()
        controller.target.actions.bias.copy_(torch.tensor(target, dtype=torch.float32))


@pytest.mark.parametrize(
    "terminal,next_mask,gamma,duration,reward,expected_target",
    [
        # Online action 2 wins among the legal actions, although target action 0 is larger.
        # Target = 1 + (1/2)^3 * 7 = 1.875. Action 1 is illegal despite both nets preferring it.
        (False, [True, False, True], 0.5, 3, 1.0, 1.875),
        # No continuation is permitted after the final graded patch.
        (True, [True, False, True], 0.99, 1, -1.0, -1.0),
        # A forced next action remains a valid bootstrap action when raw transitions are used.
        (False, [True, False, False], 0.5, 2, 1.0, 13.5),
        # Zero discount removes continuation even for a nonterminal transition.
        (False, [True, False, True], 0.0, 1, 2.0, 2.0),
    ],
)
def test_numeric_double_dqn_targets(terminal, next_mask, gamma, duration, reward, expected_target):
    cfg = settings()
    cfg.learner.gamma = gamma
    controller = Controller(cfg)
    constant_values(controller, [1, 30, 2], [50, 900, 7])
    transition = row(controller, reward=reward, terminal=terminal, next_mask=next_mask, duration=duration)
    metrics = controller.update([{"transitions": [transition]}])
    error = abs(expected_target - 1.0)
    expected_huber = 0.5 * error**2 if error <= 1 else error - 0.5
    assert metrics["loss"] == pytest.approx(expected_huber)
    assert metrics["optimizer_steps"] == 1
    assert all(parameter.grad is None for parameter in controller.target.parameters())


def test_target_copy_clock_is_optimizer_steps():
    cfg = settings()
    cfg.learner.target_interval = 2
    cfg.learner.target_clock = "updates"
    controller = Controller(cfg)
    constant_values(controller, [0, 0, 0], [0, 0, 0])
    original = copy.deepcopy(controller.target.state_dict())

    controller.update([{"transitions": [row(controller, reward=4.0)]}])
    assert controller.steps == 1
    assert all(torch.equal(value, original[key]) for key, value in controller.target.state_dict().items())
    assert not torch.equal(controller.net.actions.bias, controller.target.actions.bias)

    controller.update([{"transitions": [row(controller, reward=4.0)]}])
    assert controller.steps == 2
    assert all(
        torch.equal(value, controller.net.state_dict()[key]) for key, value in controller.target.state_dict().items()
    )
    copied = copy.deepcopy(controller.target.state_dict())

    controller.update([{"transitions": [row(controller, reward=4.0)]}])
    assert controller.steps == 3
    assert all(torch.equal(value, copied[key]) for key, value in controller.target.state_dict().items())
    assert not torch.equal(controller.net.actions.bias, controller.target.actions.bias)


def test_replay_update_budget_does_not_depend_on_episode_boundaries():
    cfg = settings()
    cfg.learner.learning_starts = 3
    cfg.learner.transitions_per_update = 2
    cfg.learner.batch_size = 4
    incremental = Controller(cfg)
    bulk = Controller(cfg)
    # Inserts 1 and 2 are warm-up. The complete post-warm-up pairs are (3,4), (5,6), (7,8).
    for count, expected_updates in [(2, 0), (2, 1), (3, 1), (2, 1)]:
        metrics = incremental.update([{"transitions": [row(incremental)]} for _ in range(count)])
        assert metrics["optimizer_steps"] == expected_updates
    bulk.update([{"transitions": [row(bulk)]} for _ in range(9)])
    assert incremental.inserted == bulk.inserted == 9
    assert incremental.steps == bulk.steps == 3
    assert len(incremental.replay) == len(bulk.replay) == 9


def test_zero_warmup_still_requires_new_experience():
    cfg = settings()
    cfg.learner.learning_starts = 0
    controller = Controller(cfg)
    empty = controller.update([{"transitions": []}])
    assert empty["optimizer_steps"] == 0
    first = controller.update([{"transitions": [row(controller)]}])
    assert first["optimizer_steps"] == 1
    unchanged = copy.deepcopy(controller.net.state_dict())
    empty = controller.update([{"transitions": []}])
    assert empty["optimizer_steps"] == 0
    assert all(torch.equal(value, unchanged[key]) for key, value in controller.net.state_dict().items())


def test_epsilon_decay_counts_genuine_training_choices_only():
    cfg = settings()
    cfg.learner.epsilon_start, cfg.learner.epsilon_end, cfg.learner.epsilon_choices = 1.0, 0.0, 4
    controller = Controller(cfg)
    constant_values(controller, [5, 1, -4], [5, 1, -4])

    class FixedDraw:
        draws = 0

        def random(self):
            self.draws += 1
            return 0.5

        def choice(self, legal):
            return legal[-1]

    controller.random = FixedDraw()
    observation = np.zeros(controller.observation_size, np.float32)
    for _ in range(20):
        assert controller.decide(observation, [True, False, False]).action == Action.CONTINUE
    assert controller.choice_points == 0

    # Epsilon before these decisions is 1, 0.75, and 0.5. A draw of 0.5 explores only the first two.
    actions = [controller.decide(observation, [True, True, False]).action for _ in range(3)]
    assert actions == [Action.PREPARE, Action.PREPARE, Action.CONTINUE]
    assert controller.choice_points == 3
    draws = controller.random.draws
    for _ in range(5):
        assert controller.decide(observation, [True, True, False], training=False).action == Action.CONTINUE
    assert controller.choice_points == 3 and controller.random.draws == draws
    assert controller.decide(observation, [True, True, False]).action == Action.CONTINUE  # epsilon = 0.25
    assert controller.decide(observation, [True, True, False]).action == Action.CONTINUE  # epsilon = 0


def recorded_episode(controller, name, mark, rewards, masks):
    transitions = []
    for i, (reward, mask) in enumerate(zip(rewards, masks, strict=True)):
        transition = row(controller, reward=reward, mask=mask, mark=mark + i / 10, terminal=i == len(rewards) - 1)
        decision = controller.decide(transition["observation"], mask)
        transition.update(action=int(decision.action), version=decision.version)
        transitions.append(transition)
    for current, following in zip(transitions, transitions[1:], strict=False):
        current.update(next_observation=following["observation"], next_mask=following["mask"])
    return {"name": name, "transitions": transitions}


def test_checkpoint_and_rebuilt_replay_produce_the_same_next_update(tmp_path):
    cfg = settings()
    cfg.learner.hidden = [8]
    cfg.learner.choice_only = True
    cfg.learner.gamma = 0.9
    cfg.learner.replay_capacity = 5
    cfg.learner.batch_size = 3
    cfg.learner.transitions_per_update = cfg.learner.target_interval = 2
    uninterrupted = Controller(cfg)
    names = ["finished-b", "finished-a", "finished-c"]  # completion order intentionally differs from filename order
    episodes = tmp_path / "episodes"
    episodes.mkdir()
    for i, name in enumerate(names):
        episode = recorded_episode(
            uninterrupted,
            name,
            i / 3,
            [-0.1, -0.2, -0.1, float(i % 2 == 0)],
            [[True, False, False], [True, True, False], [True, False, False], [True, False, True]],
        )
        with gzip.open(episodes / f"{name}.json.gz", "wt") as stream:
            json.dump(episode, stream)
        uninterrupted.update([episode])
    uninterrupted.save(tmp_path / "policy.pt", metadata={"learned_names": names, "updates": []})
    resumed = Controller(cfg)
    resumed.load(tmp_path / "policy.pt", full=True)
    assert rebuild_replay(tmp_path, resumed.checkpoint_metadata["learned_names"], resumed) == 5
    assert resumed.inserted == uninterrupted.inserted == 6
    assert resumed.steps == uninterrupted.steps == 3
    assert resumed.choice_points == uninterrupted.choice_points == 6
    for original, recovered in zip(uninterrupted.replay, resumed.replay, strict=True):
        assert original.keys() == recovered.keys()
        for key in original:
            np.testing.assert_array_equal(original[key], recovered[key])

    def next_episode(controller):
        return recorded_episode(
            controller,
            "next",
            0.4,
            [-0.3, 1.0],
            [[True, True, False], [True, False, True]],
        )

    fresh_original, fresh_resumed = next_episode(uninterrupted), next_episode(resumed)
    assert fresh_original == fresh_resumed  # includes the next epsilon-greedy draws
    result_original, result_resumed = uninterrupted.update([fresh_original]), resumed.update([fresh_resumed])
    assert result_original["loss"] == pytest.approx(result_resumed["loss"], abs=1e-8)
    assert (uninterrupted.version, uninterrupted.steps, uninterrupted.choice_points) == (
        resumed.version,
        resumed.steps,
        resumed.choice_points,
    )
    for first, second in [(uninterrupted.net, resumed.net), (uninterrupted.target, resumed.target)]:
        assert all(torch.equal(value, second.state_dict()[key]) for key, value in first.state_dict().items())
