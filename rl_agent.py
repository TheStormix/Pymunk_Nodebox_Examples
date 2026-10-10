# -*- coding: utf-8 -*-
"""Tabular Q-learning for the robot in 8_6.py (Python 2.7 and Python 3)."""
from __future__ import division

import random


class QLearningAgent(object):
    """Learn Q(state, action) using the one-step temporal-difference update.

    States: 0=nothing, 1=target, 2=anti-target, 3=boundary.
    Actions: 0=keep heading, 1=change heading.
    """

    def __init__(self, states=4, actions=2, alpha=0.30, gamma=0.90,
                 epsilon=0.60, epsilon_min=0.05, epsilon_decay=0.999,
                 seed=9):
        if states < 1 or actions < 1:
            raise ValueError('states and actions must be positive')
        if not (0 < alpha <= 1 and 0 <= gamma <= 1):
            raise ValueError('invalid alpha/gamma')
        if not (0 <= epsilon_min <= epsilon <= 1):
            raise ValueError('invalid epsilon range')
        if not 0 < epsilon_decay <= 1:
            raise ValueError('invalid epsilon decay')
        self.q = [[0.0 for _ in range(actions)] for _ in range(states)]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)
        self.previous_state = None
        self.previous_action = None
        self.steps = 0
        self.updates = 0

    def _check_state(self, state):
        if type(state) is not int or not 0 <= state < len(self.q):
            raise ValueError('invalid state')

    def choose_action(self, state):
        self._check_state(state)
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(len(self.q[state]))
        values = self.q[state]
        best_value = max(values)
        # Random tie breaking avoids favoring action 0 on an unseen state.
        best_actions = [i for i, value in enumerate(values) if value == best_value]
        return self.rng.choice(best_actions)

    def step(self, state, reward):
        """Observe the reward of the previous action; choose an action now.

        Q(s_prev,a_prev) <- Q + alpha*(r+gamma*max_a Q(s,a)-Q)
        The first call chooses an action without updating a Q-value.
        """
        self._check_state(state)
        if self.previous_state is not None:
            q_old = self.q[self.previous_state][self.previous_action]
            target = float(reward) + self.gamma * max(self.q[state])
            self.q[self.previous_state][self.previous_action] = (
                q_old + self.alpha * (target - q_old)
            )
            self.updates += 1
        action = self.choose_action(state)
        self.previous_state = state
        self.previous_action = action
        self.steps += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return action

    def finish_episode(self, reward):
        """Update a terminal state without future-reward bootstrapping."""
        if self.previous_state is not None:
            old = self.q[self.previous_state][self.previous_action]
            self.q[self.previous_state][self.previous_action] = (
                old + self.alpha * (float(reward) - old)
            )
            self.updates += 1
        self.previous_state = None
        self.previous_action = None

    def greedy_action(self, state):
        """Deterministic best known action for evaluation (does not explore)."""
        self._check_state(state)
        values = self.q[state]
        return values.index(max(values))
