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
import gymnasium as gym
import numpy as np
# from ppo_wrapper import PPOObsWrapper
import argparse
import logging
import time

from collections import deque

#TRAIN PARAMS
TOTAL_TIMESTEPS = 10_000_000
EVAL_INTERVAL = 50_000       # timesteps per evaluation chunk
TIMESTEPS_DONE = 0
NUM_ENVS = 10
# 50% self play against last snapshot, 50% random opponents
CHECKPOINT_ABSOLUTE = "/tmp/train/recurrentppo_1lr_checkpoint.zip"
RECURRENTPPO = RecurrentPPOInferenceModel("train/recurrentppo_1lr_checkpoint.zip")
RANDOM = RandomPolicyInferenceModel(ClashRoyaleGymEnv())
OPPONENT_POLICIES = [
    # RecurrentPPOInferenceModel("/tmp/train/recurrentppo_1lr_checkpoint.zip"),
   RECURRENTPPO,
   RECURRENTPPO,
   RECURRENTPPO,
   RANDOM,
   RANDOM

]
LEARNING_RATE = 1e-3


def evaluate_model(model, eval_env, n_eval_episodes=30):
    rewards = []
    episode_lengths = []

    for _ in range(n_eval_episodes):
        obs = eval_env.reset()          # VecEnv reset -> obs only
        state = None                    # LSTM hidden state
        episode_start = np.ones((eval_env.num_envs,), dtype=bool)

        total_reward = 0.0
        steps = 0
        done = False

        #

        # initialize per-evaluation wins counter on first episode
        if len(rewards) == 0:
            wins = 0

        while not done:
            action, state = model.predict(
            obs,
            state=state,
            episode_start=episode_start,
            deterministic=True
            )

            obs, reward, dones, infos = eval_env.step(action)

            total_reward += float(reward[0])
            steps += 1
            done = bool(dones[0])

            # episode_start must be True only on reset
            episode_start = dones

        # determine win/loss from final infos (VecEnv returns list)
        final_info = infos[0] if isinstance(infos, (list, tuple)) else infos
        players = final_info.get("players", []) if isinstance(final_info, dict) else []
        if len(players) >= 2:
            # sum tower HPs as a simple end-state score (player_id 0 = agent)
            hp_map = {p.get("player_id"): sum(float(x) for x in (p.get("king_hp", 0.0), p.get("left_hp", 0.0), p.get("right_hp", 0.0))) for p in players}
            our_hp = hp_map.get(0, 0.0)
            opp_hp = hp_map.get(1, 0.0)
            if our_hp > opp_hp:
                wins += 1

        # if this was the last evaluation episode, print wins
        if len(rewards) == n_eval_episodes - 1:
            print(f"Wins: {wins}/{n_eval_episodes}")

        rewards.append(total_reward)
        episode_lengths.append(steps)

    avg_reward = float(np.mean(rewards))
    avg_episode_length = float(np.mean(episode_lengths))

    print(f"Average episode length: {avg_episode_length:.2f} steps")
    return avg_reward

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
        for i in range(NUM_ENVS)
    ]
    vec_env = SubprocVecEnv(env_fns)
    eval_env = DummyVecEnv([lambda: PPOObsWrapper(PPORewardWrapper(ClashRoyaleGymEnv()))])
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
            learning_rate=1e-4,
            n_steps=8192,
            batch_size=256,
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

        avg_reward = evaluate_model(model, eval_env)

        # TIME CHECK
        if time.time() - start_time >= MAX_RUNTIME:
            print("Max runtime reached, stopping training.")
            break

        reward_history.append(avg_reward)
        print(f"Evaluation after {timesteps_done} steps: avg_reward={avg_reward:.3f}")

        if avg_reward > best_reward + 1e-3:
            best_reward = avg_reward
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"Reward plateau detected (avg_reward={best_reward:.3f}). Stopping training.")
            break

        print("Total timesteps done:", timesteps_done)

    print("Training completed.")
    print(f"Total timesteps trained: {timesteps_done}")
    print(f"Best average reward during evaluation: {best_reward:.3f}")
    print("Run time (seconds): {:.2f}".format(time.time() - start_time))

    # Save model
    model.save(f"{args.output_dir}/recurrentppo_1lr_checkpoint")
    print("Training finished, model saved.")

