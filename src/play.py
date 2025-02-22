import gymnasium as gym
import ale_py
import shimmy
import torch
import argparse

from common import DQN, preprocess

def play(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("ALE/Pong-v5", render_mode="human", frameskip=4)
    n_actions = env.action_space.n
    input_shape = (1, 84, 84)

    policy_net = DQN(input_shape, n_actions).to(device)

    # Load a saved policy network
    policy_net.load_state_dict(torch.load(f"checkpoints/policy_net_{args.frame_count}.pth", map_location=device))
    policy_net.eval()

    state, _ = env.reset()
    state = preprocess(state)

    done = False
    total_reward = 0
    while not done:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = policy_net(state_tensor)
            action = int(q_values.argmax(dim=1).item())

        next_state, reward, terminated, truncated, _ = env.step(action)
        state = preprocess(next_state)
        total_reward += reward
        done = terminated or truncated

    print(f"Total Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_count", type=str, required=True, help="Frames to load.")
    args = parser.parse_args()
    play(args)
