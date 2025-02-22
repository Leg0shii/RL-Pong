import pickle
import gymnasium as gym
import ale_py
import shimmy
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import trange
import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

from common import DQN, ReplayBuffer, preprocess

def train(args) -> None:
    """
    python src/train.py --frame_count 10000
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=4)
    n_actions = env.action_space.n
    input_shape = (1, 84, 84)

    policy_net = DQN(input_shape, n_actions).to(device)
    target_net = DQN(input_shape, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    replay_buffer = ReplayBuffer(100000)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

    # Load checkpoint if provided
    frame_start = 0
    if args.frame_count:
        frame_start = args.frame_count
        policy_net.load_state_dict(torch.load(f"checkpoints/policy_net_{frame_start}.pth"))
        print(f"\nLoaded policy_net with frame_count: {frame_start}")

        with open(f"checkpoints/replay_buffer_{frame_start}.pkl", "rb") as f:
            replay_buffer = pickle.load(f)
        print(f"Loaded replay buffer from {frame_start}")

    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 1e-6
    gamma = 0.99
    batch_size = 32
    target_update_freq = 1000
    max_frames = 5000000
    episode_rewards = []
    losses = []

    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_prefix = (f"epsilon-{epsilon}_epsilonmin-{epsilon_min}_epsilondecay-{epsilon_decay}_gamma-{gamma}"
                     f"_batchsize-{batch_size}_targetupdatefreq-{target_update_freq}_maxframes-{max_frames}")

    state, _ = env.reset()
    state = preprocess(state)
    episode_start_time = time.time()
    episode = 1
    episode_reward = 0

    checkpoint_interval = 100000
    for frame in trange(frame_start + 1, max_frames + 1):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
            action = int(q_values.argmax(dim=1).item())

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_proc = preprocess(next_state)
        replay_buffer.push(state, action, reward, next_proc, done)
        state = next_proc
        episode_reward += reward

        if done:
            episode_duration = time.time() - episode_start_time
            print(f"\nEpisode {episode} | Reward: {episode_reward} | Frame: {frame} | Time: {episode_duration:.2f}s")
            episode_rewards.append(episode_reward)
            state, _ = env.reset()
            state = preprocess(state)
            episode_reward = 0
            episode += 1
            episode_start_time = time.time()

        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            states = torch.tensor(states, dtype=torch.float32, device=device)
            actions = torch.tensor(actions, dtype=torch.long, device=device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
            next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
            dones = torch.tensor(dones, dtype=torch.float32, device=device)

            q_current = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                q_next = target_net(next_states).max(dim=1)[0]
            q_target = rewards + gamma * q_next * (1 - dones)

            loss = nn.MSELoss()(q_current, q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        epsilon = max(epsilon_min, epsilon - epsilon_decay)

        if frame % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if frame % checkpoint_interval == 0:
            torch.save(policy_net.state_dict(), f"checkpoints/policy_net_{frame}.pth")
            with open(f"checkpoints/replay_buffer_{frame}.pkl", "wb") as f:
                pickle.dump(replay_buffer, f)
            print(f"\nSaved checkpoint at frame {frame}.")

    env.close()

    torch.save(policy_net.state_dict(), f"networks/{start_time}+{config_prefix}_dqn.pth")

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
    plt.savefig(f"plots/{start_time}+{config_prefix}_training-plots.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_count", type=int, default=None, help="Frame to load policy and buffer from.")
    args = parser.parse_args()
    train(args)
