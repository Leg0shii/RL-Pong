# Reinforcement Learning with Pong

This repository contains code and documentation for training a Deep Q-Network (DQN) agent to play the Atari 2600 game **Pong** using **OpenAI Gym**. The approach follows the methodology introduced by Mnih et al. (2015), with minor adjustments to hyperparameters and environment wrappers.

## Overview

The project demonstrates how a convolutional neural network can learn to control the Pong paddle directly from preprocessed game frames without manual feature engineering. Key components include:

- **Deep Q-Learning**: A value-based reinforcement learning method approximating the action-value function via a neural network.
- **Convolutional Neural Network (CNN)**: Extracts latent features from stacks of preprocessed frames.
- **Experience Replay**: Stores transitions in a replay buffer to break temporal correlations in training samples and stabilize learning.
- **Target Network**: Improves training stability by periodically updating a separate network to compute target Q-values.

## Gameplay
View a full game here: https://youtu.be/10e7ZIhSqiY

## Main Files

- **common.py**  
  Defines the **DQN** class implementing a convolutional neural network for Q-value estimation, as well as a **ReplayBuffer** class for storing and sampling agent experiences.

- **train.py**  
  Contains the main training script. It initializes the Pong environment, creates and trains the policy network, and periodically updates a target network. Key components include:
  - Environment wrappers for preprocessing frames (grayscale, downsampling, stacking).
  - Epsilon-greedy exploration strategy for balancing exploration and exploitation.
  - RMSprop optimizer with a specified learning rate, batch size, and replay buffer capacity.

- **play.py**  
  Allows you to play the trained network that is provided inside of networks.

## References

- Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

---

For any inquiries or suggestions, feel free to open an issue or submit a pull request.
