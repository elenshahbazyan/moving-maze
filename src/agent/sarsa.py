# FILE: src/agent/sarsa.py
"""
Tabular SARSA agent (unchanged except minor improvements):
 - epsilon tie-breaking handles multiple argmax values by selecting randomly among them
 - default alpha adjusted for stability on larger grids
"""

import numpy as np
import random
from typing import Tuple

class SARSAAgent:
    """Simple tabular SARSA agent for discrete grid states.

    State mapping used here: state_id = x + y * size (+ optional time phase multiplier).
    """
    def __init__(self, size: int, n_actions: int = 5, time_mod: int = 1, alpha: float = 0.25, gamma: float = 0.99):
        self.size = int(size)
        self.n_actions = int(n_actions)
        self.time_mod = max(1, int(time_mod))
        self.alpha = float(alpha)
        self.gamma = float(gamma)

        self.n_states = self.size * self.size * self.time_mod
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float32)

    def state_id(self, agent_pos: Tuple[int, int], step_count: int = 0) -> int:
        ax, ay = int(agent_pos[0]), int(agent_pos[1])
        base = ax + ay * self.size
        if self.time_mod == 1:
            return base
        phase = int(step_count % self.time_mod)
        return base * self.time_mod + phase

    def act_epsilon(self, s: int, eps: float) -> int:
        if random.random() < eps:
            return random.randrange(self.n_actions)
        # tie-break randomly among best actions
        best = np.flatnonzero(self.Q[s] == np.max(self.Q[s]))
        return int(np.random.choice(best))

    def update(self, s: int, a: int, r: float, s2: int, a2: int) -> None:
        self.Q[s, a] += self.alpha * (r + self.gamma * self.Q[s2, a2] - self.Q[s, a])

    def save(self, path: str) -> None:
        np.save(path, self.Q)

    def load(self, path: str) -> None:
        self.Q = np.load(path)

    def greedy_action(self, s: int) -> int:
        best = np.flatnonzero(self.Q[s] == np.max(self.Q[s]))
        return int(np.random.choice(best))