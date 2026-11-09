# BEGIN AI ACKNOWLEDGMENT
# AI-assisted project preparation: OpenAI ChatGPT / Codex.
# Scope and disclosure: report/REPORT.md, Section 10.
# Detailed assistance record: report/AI_ASSISTANCE.md (Appendix C).
# Project repository: https://github.com/kankanalaramya/AI-2
# Conversation links omitted for privacy; see the disclosure in Section 10.
# END AI ACKNOWLEDGMENT
"""Focused regression checks. Run: python -m unittest -v test_algorithms.py.

AI assistance: ChatGPT helped write these checks; cite the conversation with
other AI-assisted code in the final submission.
"""

import unittest
from unittest.mock import Mock, patch
import numpy as np

from warehouse_gridworld_domain_random import WarehouseGridWorld, StepResult, ACTIONS
from value_iteration import one_outcome, state_list, terminal_state, value_iteration
from q_learning import train_q_learning, evaluate_fixed_policy


class AlgorithmTests(unittest.TestCase):
    def test_planner_matches_environment(self):
        world = WarehouseGridWorld(seed=42)
        for state in state_list(world):
            if terminal_state(state, world):
                continue
            for action in ACTIONS:
                expected = one_outcome(world, state, action)
                world.reset()
                world.robot_pos = state[:2]
                world.has_package = bool(state[2])
                world.delivered = bool(state[3])
                world.np_random = Mock()
                world.np_random.choice.return_value = action
                result = world.step(action, intended_probability=1.0)
                self.assertEqual((result.state, result.reward, result.done), expected)

    def test_timeout_keeps_future_value_but_success_does_not(self):
        # A controlled one-state experience distinguishes the two targets:
        # after learning value 1, a timeout target is 1 + 0.5 * 1 = 1.5.
        # A successful terminal transition still has target 1.
        for event, expected in [("max_steps", 1.5), ("return_to_dock", 1.0)]:
            world = WarehouseGridWorld(seed=42, max_steps=1)
            state = world.get_state()
            world.step = Mock(return_value=StepResult(state, 1.0, True, {"event": event}))
            with patch('q_learning.choose_action', return_value=0):
                q, rewards, successes, epsilons = train_q_learning(
                    world, 1.0, seed=11, episodes=2, alpha=1.0, gamma=0.5
                )
            self.assertEqual(q[state][0], expected)
            self.assertEqual(list(successes), [int(event == "return_to_dock")] * 2)

    def test_training_seed_is_independent_of_earlier_evaluation(self):
        world_a = WarehouseGridWorld(seed=42)
        world_b = WarehouseGridWorld(seed=42)
        policy = {state: "up" for state in state_list(world_b)}
        evaluate_fixed_policy(world_b, policy, 0.8, seed=999, runs=2)
        first = train_q_learning(world_a, 0.8, seed=11, episodes=30)
        second = train_q_learning(world_b, 0.8, seed=11, episodes=30)
        for state in first[0]:
            np.testing.assert_array_equal(first[0][state], second[0][state])
        for index in (1, 2, 3):
            np.testing.assert_array_equal(first[index], second[index])

    def test_training_and_evaluation_use_world_step_limit(self):
        world = WarehouseGridWorld(seed=42, max_steps=131)
        with patch('q_learning.choose_action', return_value=ACTIONS.index("up")):
            train_q_learning(world, 1.0, seed=11, episodes=1)
        self.assertEqual(world.steps, 131)
        self.assertTrue(world.done)
        policy = {state: "up" for state in state_list(world)}
        reward, success = evaluate_fixed_policy(world, policy, 1.0, seed=11, runs=1)
        self.assertEqual(world.steps, 131)
        self.assertEqual(success, 0.0)

    def test_deterministic_vi_finishes_the_task(self):
        world = WarehouseGridWorld(seed=42)
        values, policy, iterations = value_iteration(world, 1.0)
        reward, success = evaluate_fixed_policy(world, policy, 1.0, seed=11, runs=1)
        self.assertEqual(success, 1.0)
        self.assertEqual(world.robot_pos, world.dock_pos)
        self.assertTrue(world.delivered)
        self.assertEqual(reward, 143.0)


if __name__ == '__main__':
    unittest.main()
