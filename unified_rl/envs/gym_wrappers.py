"""Gym environment wrappers and utilities."""

from typing import Optional

import gym
import numpy as np
from gym.wrappers import TimeLimit


class NormalizeObservation(gym.ObservationWrapper):
    """Wrapper that normalizes observations to have zero mean and unit variance."""

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        """Initialize the wrapper.

        Args:
            env: Gym environment to wrap
            epsilon: Small constant to avoid division by zero
        """
        super().__init__(env)
        self.epsilon = epsilon
        self.mean = np.zeros(env.observation_space.shape)
        self.var = np.ones(env.observation_space.shape)
        self.count = 0

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize the observation.

        Args:
            obs: Raw observation

        Returns:
            Normalized observation
        """
        self.count += 1
        self.mean = self.mean + (obs - self.mean) / self.count
        self.var = self.var + ((obs - self.mean) ** 2 - self.var) / self.count
        return (obs - self.mean) / np.sqrt(self.var + self.epsilon)


def make_env(
    env_name: str, max_steps: int, seed: int, normalize_obs: bool = True
) -> gym.Env:
    """Create and configure a Gym environment.

    Args:
        env_name: Name of the Gym environment
        max_steps: Maximum number of steps per episode
        seed: Random seed for the environment
        normalize_obs: Whether to normalize observations

    Returns:
        Configured Gym environment
    """
    env = gym.make(env_name)
    env = TimeLimit(env, max_episode_steps=max_steps)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    if normalize_obs:
        env = NormalizeObservation(env)

    return env
