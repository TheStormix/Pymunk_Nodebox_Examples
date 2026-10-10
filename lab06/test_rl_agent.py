# -*- coding: utf-8 -*-
"""Unit tests for the Q-learning part of Lab 06 (no NodeBox required)."""
from __future__ import print_function
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rl_agent import QLearningAgent


class TestQLearningAgent(unittest.TestCase):
    def test_initial_table(self):
        a = QLearningAgent()
        self.assertEqual(len(a.q), 4)
        self.assertEqual(a.q[2], [0.0, 0.0])

    def test_first_step_has_no_update(self):
        a = QLearningAgent(epsilon=0, epsilon_min=0)
        action = a.step(0, 50)
        self.assertIn(action, (0, 1))
        self.assertEqual(a.updates, 0)
        self.assertEqual(a.q[0], [0.0, 0.0])

    def test_bellman_update_uses_previous_state_and_next_value(self):
        a = QLearningAgent(alpha=0.5, gamma=0.9, epsilon=0, epsilon_min=0)
        old_action = a.step(0, 0)
        a.q[1][:] = [2.0, 1.0]
        a.step(1, 4.0)
        self.assertAlmostEqual(a.q[0][old_action], 0.5 * (4 + 0.9 * 2))
        self.assertEqual(a.updates, 1)

    def test_exploration_floor(self):
        a = QLearningAgent(epsilon=0.6, epsilon_min=0.1, epsilon_decay=0.1)
        for i in range(20):
            a.step(0, 0)
        self.assertAlmostEqual(a.epsilon, 0.1)

    def test_no_exploration_selects_maximum(self):
        a = QLearningAgent(epsilon=0, epsilon_min=0)
        a.q[2][:] = [0.2, 1.4]
        self.assertEqual(a.choose_action(2), 1)
        self.assertEqual(a.greedy_action(2), 1)

    def test_terminal_update_and_reset(self):
        a = QLearningAgent(alpha=0.5, gamma=1, epsilon=0, epsilon_min=0)
        action = a.step(0, 0)
        a.finish_episode(2)
        self.assertEqual(a.q[0][action], 1.0)
        self.assertIsNone(a.previous_action)
        self.assertEqual(a.updates, 1)

    def test_fixed_seed_reproducibility(self):
        a = QLearningAgent(seed=29)
        b = QLearningAgent(seed=29)
        seq_a = [a.step(s, 0.25) for s in [0, 1, 0, 3, 2, 1] * 5]
        seq_b = [b.step(s, 0.25) for s in [0, 1, 0, 3, 2, 1] * 5]
        self.assertEqual(seq_a, seq_b)
        self.assertEqual(a.q, b.q)

    def test_invalid_state(self):
        a = QLearningAgent()
        with self.assertRaises(ValueError):
            a.step(5, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
