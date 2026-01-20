# train_ppo.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.clasher.gym_env import ClashRoyaleGymEnv
import gymnasium as gym
import numpy as np
# from ppo_wrapper import PPOObsWrapper
import argparse
import logging

#!! : MAKE SURE YOU ALLOW ARG PARSING WITH DOUBLE TAGS
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
    

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='output', help='Directory to save the trained models')
args = parser.parse_args()

# Create logs and model dirs
os.makedirs(args.output_dir, exist_ok=True)

#logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Wrap env for PPO
env = PPOObsWrapper(ClashRoyaleGymEnv())
vec_env = DummyVecEnv([lambda: env])

# Instantiate PPO

model = PPO(
    "CnnPolicy",
    vec_env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=4096,
    batch_size=256,
    n_epochs=6,
    gamma=0.997,
    gae_lambda=0.98,
    ent_coef=0.003,
    policy_kwargs=dict(normalize_images=False),
)


#log model
logger.info("Starting training...")

# Train for 100k timesteps (adjust as needed)
model.learn(total_timesteps=100_000, progress_bar=True)

logger.info("Training completed.")

# Save model
model.save(f"{args.output_dir}/ppo_clashroyale_baseline")
logger.info("Training finished, model saved.")