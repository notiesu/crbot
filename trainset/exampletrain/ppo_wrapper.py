# environment wrapper for PPO

# train_ppo.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.clasher.gym_env import ClashRoyaleGymEnv
import gymnasium as gym
import numpy as np

class PPOObsWrapper(gym.ObservationWrapper):
    """
    Wraps ClashRoyaleGymEnv for PPO.
    Converts dict obs to normalized Box tensor.
    """
    def __init__(self, env: ClashRoyaleGymEnv):
        super().__init__(env)
        self.env = env  # Explicitly set the environment
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, 128, 128),  # channel-first
            dtype=np.float32
        )
        self.action_space = self.env.action_space

    def observation(self, obs):
        # normalize 0-255 -> 0-1
        img = obs["p1-view"].astype(np.float32)
        # convert HWC -> CHW
        return np.transpose(img, (2, 0, 1))