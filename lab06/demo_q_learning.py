# -*- coding: utf-8 -*-
"""A discrete Q-learning sanity check, not the Pymunk physics simulator.

Observations are sampled from four sensor states with known reward preferences.
The test shows that tabular Q-learning can learn the preferred action.
"""
from __future__ import print_function
import csv
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rl_agent import QLearningAgent


def train(steps=10000, seed=9):
    agent = QLearningAgent(seed=seed)
    rng = random.Random(seed + 1)
    # Desired action: when target is visible, keep the heading; when
    # anti-target or boundary is visible, change it.
    rewards = ((0.0, 0.0), (1.0, -1.0), (-1.0, 1.0), (-0.5, 1.0))
    history = []
    previous_action = None
    previous_state = None
    for i in range(steps):
        state = rng.randrange(4)
        reward = rewards[previous_state][previous_action] if previous_action is not None else 0.0
        action = agent.step(state, reward)
        previous_state = state
        previous_action = action
        if (i + 1) % 1000 == 0:
            history.append((i + 1, agent.epsilon,
                            agent.greedy_action(1), agent.greedy_action(2),
                            agent.greedy_action(3)))
    return agent, history


def main():
    a, history = train()
    folder = os.path.join(os.path.dirname(__file__), 'results')
    if not os.path.isdir(folder):
        os.makedirs(folder)
    path = os.path.join(folder, 'q_learning_training.csv')
    with open(path, 'w') as f:
        w = csv.writer(f)
        w.writerow(['step', 'epsilon', 'target_action', 'anti_target_action', 'boundary_action'])
        for row in history:
            w.writerow(row)
    labels = ('nothing', 'target', 'anti-target', 'boundary')
    print('Q-learning: %d decisions, %d TD updates' % (a.steps, a.updates))
    for state, label in enumerate(labels):
        print('%-12s %s -> action %d' % (label, ['%.3f' % v for v in a.q[state]], a.greedy_action(state)))
    print('Saved:', path)
    # This expectation verifies a synthetic reward model, NOT success in the
    # actual graphical simulation (which requires NodeBox/Pymunk on Python 2.7).
    assert [a.greedy_action(i) for i in (1, 2, 3)] == [0, 1, 1]
    print('Synthetic learning check: PASS')


if __name__ == '__main__':
    main()
