import csv
import json
import os
import pickle
import random
import torch

import ale_py
import shimmy

import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from collections import deque
from datetime import datetime
from gymnasium.vector import AsyncVectorEnv
from matplotlib import pyplot as plt
from tqdm import trange

import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
from gymnasium.spaces import Box

from common import DQN, ReplayBuffer

# --- No-Op and Fire Wrapper ---
class NoopAndFireEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max+1)
        for _ in range(noops):
            obs, _, done, truncated, info = self.env.step(self.noop_action)
            if done or truncated:
                obs, info = self.env.reset(**kwargs)
        obs, _, done, truncated, info = self.env.step(1)
        if done or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info

# --- Wrapper für Graustufen ---
class PreprocessWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8
        )
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process(obs), info
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return self._process(obs), reward, done, truncated, info
    def _process(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

def make_env():
    def _thunk():
        env = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=4)
        env = NoopAndFireEnv(env)
        env = PreprocessWrapper(env)
        env = FrameStackObservation(env, 4)
        return env
    return _thunk

def create_output_dirs(base_dir="outputs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, timestamp)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    return output_dir, checkpoints_dir, plots_dir, timestamp

def save_config(config, output_dir, timestamp):
    with open(os.path.join(output_dir, f"config_{timestamp}.json"), "w") as f:
        json.dump(config, f, indent=4)

def save_checkpoint(policy_net, replay_buffer, frame, checkpoints_dir):
    torch.save(
        policy_net.state_dict(),
        os.path.join(checkpoints_dir, f"policy_net_{frame}.pth")
    )

def save_rewards(episode_rewards, output_dir):
    with open(os.path.join(output_dir, "episode_rewards.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Reward"])
        writer.writerows(enumerate(episode_rewards))

def plot_metrics(episode_rewards, losses, plots_dir):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "training_plots.png"))
    plt.close()

def main():
    config = {
        "num_envs": 8,
        "max_frames": 5000000,
        "lr": 2.5e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "buffer_size": 200000,
        "target_update_freq": 1000,
        "epsilon": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 1e-6
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir, checkpoints_dir, plots_dir, timestamp = create_output_dirs()
    save_config(config, output_dir, timestamp)

    envs = AsyncVectorEnv([make_env() for _ in range(config["num_envs"])])
    n_actions = envs.single_action_space.n
    input_shape = (4, 84, 84)

    policy_net = DQN(input_shape, n_actions).to(device)
    target_net = DQN(input_shape, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    replay_buffer = ReplayBuffer(config["buffer_size"])

    optimizer = optim.RMSprop(policy_net.parameters(), 
                              lr=config["lr"],
                              alpha=0.95,
                              eps=0.01)

    obs, _ = envs.reset()
    total_rewards = np.zeros(config["num_envs"], dtype=np.float32)
    episode_rewards = []
    losses = []
    epsilon = config["epsilon"]
    last_print = 0

    def handle_checkpoint(frame):
        save_checkpoint(policy_net, replay_buffer, frame, checkpoints_dir)
        print(f"\nCheckpoint saved at frame {frame}.")

    try:
        for frame in trange(1, config["max_frames"] + 1):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                q_values = policy_net(obs_t)
            actions = []
            for i in range(config["num_envs"]):
                if np.random.rand() < epsilon:
                    actions.append(envs.single_action_space.sample())
                else:
                    actions.append(int(q_values[i].argmax().item()))
            next_obs, rewards, dones, truncs, _ = envs.step(actions)
            for i in range(config["num_envs"]):
                replay_buffer.push(obs[i], actions[i], rewards[i], 
                                   next_obs[i], dones[i] or truncs[i])
                total_rewards[i] += rewards[i]
                if dones[i] or truncs[i]:
                    episode_rewards.append(total_rewards[i])
                    total_rewards[i] = 0.0
            obs = next_obs
            if len(replay_buffer) >= config["batch_size"]:
                s_b, a_b, r_b, ns_b, d_b = replay_buffer.sample(config["batch_size"])
                s_v = torch.tensor(s_b, dtype=torch.float32, device=device)
                ns_v = torch.tensor(ns_b, dtype=torch.float32, device=device)
                a_v = torch.tensor(a_b, dtype=torch.long, device=device)
                r_v = torch.tensor(r_b, dtype=torch.float32, device=device)
                d_v = torch.tensor(d_b, dtype=torch.float32, device=device)

                q_current = policy_net(s_v).gather(1, a_v.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_actions = policy_net(ns_v).argmax(dim=1, keepdim=True)
                    q_next = target_net(ns_v).gather(1, next_actions).squeeze(1)
                q_target = r_v + config["gamma"] * q_next * (1 - d_v)

                # Huber loss
                loss = nn.SmoothL1Loss()(q_current, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            epsilon = max(config["epsilon_min"], epsilon - config["epsilon_decay"])
            if frame % config["target_update_freq"] == 0:
                target_net.load_state_dict(policy_net.state_dict())
            if frame % 200000 == 0:
                handle_checkpoint(frame)

            current_episodes = len(episode_rewards)
            if current_episodes % 100 == 0 and current_episodes > last_print:
                print(f"Average 100 Episode Rewards: {np.mean(episode_rewards[-100:])}")
                last_print = current_episodes

    except KeyboardInterrupt:
        print("\nTraining stopped. Saving...")

    finally:
        envs.close()
        save_rewards(episode_rewards, output_dir)
        plot_metrics(episode_rewards, losses, plots_dir)
        print(f"Done. Data in: {output_dir}")

if __name__ == "__main__":
    main()
