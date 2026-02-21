# train_ppo.py
# Install sb3_contrib if not already installed
from asyncio.log import logger
import subprocess
import sys
from sb3_contrib import RecurrentPPO
import os
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from wrappers.rppowrappers import PPOObsWrapper, PPORewardWrapper


from src.clasher.gym_env import ClashRoyaleGymEnv, ClashRoyaleVectorEnv

from wrappers.recurrentppo import RecurrentPPOInferenceModel
from wrappers.randompolicy import RandomPolicyInferenceModel
from wrappers.replaymodel import ReplayInferenceModel
from eval_env import EvalVectorEnv
import gymnasium as gym
import numpy as np
# from ppo_wrapper import PPOObsWrapper

import argparse
import logging
import time

from collections import deque

#training against a scripted opponent - run evaluations 
TOTAL_TIMESTEPS = 10_000_000
EVAL_INTERVAL = 50_000       # timesteps per evaluation chunk
TIMESTEPS_DONE = 0
NUM_ENVS_TRAINING = 10
NUM_ENVS_EVAL = 5
N_STEPS = 1024
BATCH_SIZE = 512
# 50% self play against last snapshot, 50% random opponents
# RECURRENTPPO = RecurrentPPOInferenceModel("train/recurrentppo_1lr_checkpoint.zip")
# RANDOM = RandomPolicyInferenceModel(ClashRoyaleGymEnv())
MODEL_NAME = "RPPO_ScriptedOpponent"
CHECKPOINT_ABSOLUTE = f"/tmp/train/{MODEL_NAME}.zip"
OPPONENT = ReplayInferenceModel(ClashRoyaleGymEnv(), replay_path="/tmp/train/replay_opponent.jsonl")
OPPONENT_POLICIES = [
    # RecurrentPPOInferenceModel("/tmp/train/recurrentppo_1lr_checkpoint.zip"),
    OPPONENT
]
LEARNING_RATE = 1e-3

patience = 5                  # stop if no improvement for 5 evals
best_reward = -np.inf
no_improve_count = 0
reward_history = deque(maxlen=patience)
timesteps_done = 0

if __name__ == "__main__":

    
    MAX_RUNTIME = int(18 * 60)  # Maximum runtime in seconds (18 minutes)
    #start timer
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save the trained models')
    parser.add_argument('--run_name', type=str, default='', help='Name of the current run')
    parser.add_argument('--train_dir', type=str, default='', help='Directory for training data')
    args = parser.parse_args()
    print("Current working directory:", os.getcwd())
    os.makedirs(args.output_dir, exist_ok=True)

    #Instantiate vec_env with the proper wrappers
    
    def make_env(opponent_policy):
        def _init():
            env = ClashRoyaleGymEnv()
            env.set_opponent_policy(opponent_policy)
            env = PPOObsWrapper(env)
            env = PPORewardWrapper(env)
            return env
        return _init

    env_fns = [
        make_env(OPPONENT_POLICIES[i % len(OPPONENT_POLICIES)])
        for i in range(NUM_ENVS_TRAINING)
    ]
    vec_env = SubprocVecEnv(env_fns)
    eval_env = EvalVectorEnv(num_envs=NUM_ENVS_EVAL, opponent_policies=OPPONENT_POLICIES)
    # Instantiate PPO
    if os.path.exists(CHECKPOINT_ABSOLUTE):
        print("Loading checkpoint...")
        model = RecurrentPPO.load(CHECKPOINT_ABSOLUTE, env=vec_env)
        
        timesteps_done = model.num_timesteps
    else:
        print("Creating new model...")
        model = RecurrentPPO(
            "MlpLstmPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            gamma=0.997,
            gae_lambda=0.98,
            ent_coef=0.002,
        )

    model.lr_schedule = lambda _: LEARNING_RATE

    
    print(f"type(model): {type(model)}")
    print(f"model.policy: {model.policy}")

    print("Starting training...")

    model.verbose = 1

    while timesteps_done < TOTAL_TIMESTEPS and (time.time() - start_time < MAX_RUNTIME):
        chunk = min(EVAL_INTERVAL, TOTAL_TIMESTEPS - timesteps_done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
        timesteps_done += chunk

        # TIME CHECK
        if time.time() - start_time >= MAX_RUNTIME:
            print("Max runtime reached, stopping training.")
            break
        
        # Evaluate
        print("Evaluating current model...")
        #wrap the model in an inference wrapper to handle action masks and recurrent states
        eval_model = RecurrentPPOInferenceModel()
        eval_model.model = model
        eval_env.evaluate(eval_model, num_episodes=30)

        # TIME CHECK
        if time.time() - start_time >= MAX_RUNTIME:
            print("Max runtime reached, stopping training.")
            break

        print("Total timesteps done:", timesteps_done)

    print("Training completed.")
    print(f"Total timesteps trained: {timesteps_done}")
    print(f"Best average reward during evaluation: {best_reward:.3f}")
    print("Run time (seconds): {:.2f}".format(time.time() - start_time))

    # Save model
    model.save(f"{args.output_dir}/{MODEL_NAME}")
    print("Training finished, model saved.")

