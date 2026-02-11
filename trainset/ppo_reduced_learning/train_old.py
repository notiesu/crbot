# train_ppo.py
# Install sb3_contrib if not already installed
import subprocess
import sys
from sb3_contrib import RecurrentPPO
import os
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback


from src.clasher.gym_env import ClashRoyaleGymEnv
import gymnasium as gym
import numpy as np
# from ppo_wrapper import PPOObsWrapper
import argparse
import logging
import time

from collections import deque

logger = configure(
    folder="./output/logs/",
    format_strings=["stdout", "tensorboard"]
)

#TRAINING PARAMS
total_timesteps = 10_000_000
eval_interval = 50_000       # timesteps per evaluation chunk
patience = 5                  # stop if no improvement for 5 evals
best_reward = -np.inf
no_improve_count = 0
reward_history = deque(maxlen=patience)
timesteps_done = 0

#SET OPPONENT MODEL PATH
opponent_model_path = "/tmp/train/recurrentppo_1lr_checkpoint.zip"


#!! : MAKE SURE YOU ALLOW ARG PARSING WITH DOUBLE TAGS


MAX_RUNTIME = int(18 * 60)  # Maximum runtime in seconds (18 minutes)
#start timer
start_time = time.time()

#Observation wrapper
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
        img = obs.astype(np.float32)
        # convert HWC -> CHW
        return np.transpose(img, (2, 0, 1))

#Reward wrapper
class PPORewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # track previous tower HPs per player: {player_id: (king,left,right)}
        self._prev_tower_hps = None
        # track whether main tower has been hit before for activation reward
        self._main_hit_seen = {0: False, 1: False}
        # elixir overflow tracking
        self._prev_elixir_waste = 0.0
        self._prev_time = 0.0
        self._elixir_overflow_accum = 0.0

        # constants for normalization
        self._H_main = 4824.0
        self._H_aux = 3631.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        shaped_reward = 0.0

        # base reward passed through (score delta)
        shaped_reward += float(reward)

        # gather current tower HPs from info (players list assumed length 2)
        players = info.get("players", [])
        cur_hps = {}
        for p in players:
            pid = p.get("player_id")
            cur_hps[pid] = (
                float(p.get("king_hp", 0.0)),
                float(p.get("left_hp", 0.0)),
                float(p.get("right_hp", 0.0)),
            )

        # initialize prev hps if missing
        if self._prev_tower_hps is None:
            self._prev_tower_hps = cur_hps.copy()
            self._prev_time = float(info.get("time", 0.0))
            self._prev_elixir_waste = float(info.get("elixir_waste", 0.0))
            return obs, shaped_reward, terminated, truncated, info

        # 1) Defensive Tower Health Reward
        r_tower = 0.0
        for pid in cur_hps:
            prev = self._prev_tower_hps.get(pid, (0.0, 0.0, 0.0))
            cur = cur_hps[pid]
            for i in range(3):
                prev_hp = prev[i]
                cur_hp = cur[i]
                delta_h = max(0.0, prev_hp - cur_hp)
                H = self._H_main if i == 0 else self._H_aux
                sign = (-1) ** (pid + 1)
                r_tower += sign * (delta_h / H)

        shaped_reward += r_tower

        # 2) Defensive Tower Destruction Reward
        r_destroy = 0.0
        for pid in cur_hps:
            prev = self._prev_tower_hps.get(pid, (0.0, 0.0, 0.0))
            cur = cur_hps[pid]
            for i in range(3):
                prev_hp = prev[i]
                cur_hp = cur[i]
                if prev_hp > 0.0 and cur_hp <= 0.0:
                    base = 3.0 if i == 0 else 1.0
                    sign = (-1) ** (pid + 1)
                    r_destroy += sign * base

        shaped_reward += r_destroy

        # 3) Main Tower Activation Reward
        r_activate = 0.0
        for pid in cur_hps:
            prev = self._prev_tower_hps.get(pid, (0.0, 0.0, 0.0))
            cur = cur_hps[pid]
            # both auxiliary towers alive (prev and cur)
            aux_prev_alive = prev[1] > 0.0 and prev[2] > 0.0
            aux_cur_alive = cur[1] > 0.0 and cur[2] > 0.0
            # main lost health for first time
            if aux_prev_alive and aux_cur_alive and not self._main_hit_seen[pid] and cur[0] < prev[0]:
                sign = (-1) ** pid
                r_activate += sign * 0.1
                self._main_hit_seen[pid] = True

        shaped_reward += r_activate

        # 4) Elixir Overflow Penalty: -0.1 per full second of continued overflow
        cur_elixir_waste = float(info.get("elixir_waste", 0.0))
        cur_time = float(info.get("time", self._prev_time))
        # consider overflow happening if total elixir_waste increased
        if cur_elixir_waste > self._prev_elixir_waste + 1e-9:
            self._elixir_overflow_accum += (cur_time - self._prev_time)

        penalty = 0.0
        if self._elixir_overflow_accum >= 1.0:
            n = int(self._elixir_overflow_accum)
            penalty = 0.1 * n
            self._elixir_overflow_accum -= n

        shaped_reward -= penalty

        # update trackers
        self._prev_tower_hps = cur_hps.copy()
        self._prev_elixir_waste = cur_elixir_waste
        self._prev_time = cur_time

        return obs, shaped_reward, terminated, truncated, info

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='output', help='Directory to save the trained models')
parser.add_argument('--run_name', type=str, default='', help='Name of the current run')
parser.add_argument('--train_dir', type=str, default='', help='Directory for training data')
args = parser.parse_args()

# Create logs and model dirs

#can i print the execution directory real quick?
print("Current working directory:", os.getcwd())
CHECKPOINT_ABSOLUTE = "/tmp/train/recurrentppo_1lr_checkpoint.zip"
os.makedirs(args.output_dir, exist_ok=True)

# Wrap env for PPO
env = PPORewardWrapper(PPOObsWrapper(ClashRoyaleGymEnv()))
vec_env = DummyVecEnv([lambda: env])

# callback for printing model params per step

from stable_baselines3.common.callbacks import BaseCallback

class PrintAllLoggerKeys(BaseCallback):
    def _on_step(self) -> bool:
        logs = self.logger.name_to_value
        if logs:  # only print if there’s something
            print(" | ".join(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" 
                             for k, v in logs.items()))
        return True


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

print(f"type(model): {type(model)}")
print(f"model.policy: {model.policy}")

# Reward plateau training loop

eval_env = DummyVecEnv([lambda: PPORewardWrapper(PPOObsWrapper(ClashRoyaleGymEnv()))])
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

#log model
logger.info("Starting training...")

model.verbose = 1
# Train for up to total_timesteps with plateau detection
while timesteps_done < total_timesteps and (time.time() - start_time < MAX_RUNTIME):
    chunk = min(eval_interval, total_timesteps - timesteps_done)
    model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False, callback=PrintAllLoggerKeys())
    timesteps_done += chunk

    #TIME CHECK
    if time.time() - start_time >= MAX_RUNTIME:
        logger.info("Max runtime reached, stopping training.")
        break
    avg_reward = evaluate_model(model, eval_env)
    
    #TIME CHECK
    if time.time() - start_time >= MAX_RUNTIME:
        logger.info("Max runtime reached, stopping training.")
        break
    reward_history.append(avg_reward)
    logger.info(f"Evaluation after {timesteps_done} steps: avg_reward={avg_reward:.3f}")

    if avg_reward > best_reward + 1e-3:
        best_reward = avg_reward
        no_improve_count = 0
    else:
        no_improve_count += 1

    if no_improve_count >= patience:
        logger.info(f"Reward plateau detected (avg_reward={best_reward:.3f}). Stopping training.")
        break
    print("Total timesteps done:", timesteps_done)

logger.info("Training completed.")
logger.info(f"Total timesteps trained: {timesteps_done}")
logger.info(f"Best average reward during evaluation: {best_reward:.3f}")
logger.info("Run time (seconds): {:.2f}".format(time.time() - start_time))

# Save model
model.save(f"{args.output_dir}/recurrentppo_1lr_checkpoint")
logger.info("Training finished, model saved.")