import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/20250225_210330/episode_rewards.csv")  # Adjust name/path as needed
rolling_mean = df["Reward"].rolling(window=100).mean()

plt.plot(df["Episode"], df["Reward"], alpha=0.4, label="Raw Reward")
plt.plot(df["Episode"], rolling_mean, color="red", label="Rolling Mean (100)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Network Improvement Over Time")
plt.legend()
plt.show()
