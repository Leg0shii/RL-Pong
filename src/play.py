import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
import cv2
from common import DQN

import ale_py
import shimmy

# Preprocess wrapper (identical to training)
class PreprocessWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
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
        env = gym.make("ALE/Pong-v5", render_mode="human", full_action_space=False)
        env = PreprocessWrapper(env)
        env = FrameStackObservation(env, stack_size=4)
        return env
    return _thunk

def load_checkpoint_and_run(checkpoint_path, episodes=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env()()  # single environment instance
    n_actions = env.action_space.n
    policy_net = DQN((4, 84, 84), n_actions).to(device)
    policy_net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    policy_net.eval()  # set evaluation mode

    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        while not (done or truncated):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_values = policy_net(obs_t)
            # print("Q-values:", q_values.cpu().numpy())
            action = int(q_values.argmax(dim=1).item())
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            # time.sleep(0.03)
        print(f"Episode {ep + 1}, Reward: {total_reward}")
        rewards.append(total_reward)
    
    print(f"Average: {np.mean(rewards)}")

if __name__ == "__main__":
    load_checkpoint_and_run("network/policy_net_5000000.pth", episodes=1)
