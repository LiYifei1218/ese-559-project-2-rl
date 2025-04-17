import random
import numpy as np
from collections import deque

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

# Assuming your RobotWorldEnv is defined as in your snippet:
import project2_env


# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# DQN Agent definition
class DQNAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim=64,
                 lr=1e-3,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.05,
                 epsilon_decay=0.99,
                 memory_capacity=10000,
                 batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_capacity)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")

        # Initialize Q-network and target network
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # target network is used only for inference

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

    def select_action(self, state, training=True):
        # Use epsilon-greedy policy
        if random.random() < self.epsilon and training:
            return random.randrange(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a random mini-batch of transitions
        transitions = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute current Q values
        current_q = self.q_network(states).gather(1, actions)

        # Compute target Q values
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # Compute loss (Mean Squared Error)
        loss = nn.MSELoss()(current_q, target_q)

        # Perform a gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon for exploration
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    # ---------- persistence ----------
    def save(self, path: str):
        torch.save(
            {
                "q_net":      self.q_network.state_dict(),
                "target_net": self.target_network.state_dict(),
                "optimizer":  self.optimizer.state_dict(),
                "epsilon":    self.epsilon,
                "step":       getattr(self, "training_steps", 0)
            },
            path
        )
        print(f"[✓] Saved checkpoint to {path}")

    def load(self, path: str, eval_mode: bool = False):
        ckpt = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(ckpt["q_net"])
        self.target_network.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = 0.0 if eval_mode else ckpt["epsilon"]
        # Switch the policy net to eval() so layers like BatchNorm / Dropout behave correctly
        self.q_network.eval() if eval_mode else self.q_network.train()
        self.target_network.eval()
        print(f"[✓] Loaded checkpoint from {path}  (eval_mode={eval_mode})")

# Training loop for the DQN agent
def train_dqn(agent, env, num_episodes=500, target_update_interval=10):
    scores = []
    for episode in range(num_episodes):
        # Reset environment; note that Gymnasium's reset returns (observation, info)
        state, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            # Gymnasium step returns: observation, reward, terminated, truncated, info
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated or (total_reward < -1000)

            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            total_reward += reward

        scores.append(total_reward)

        # Update target network periodically for better stability.
        if episode % target_update_interval == 0:
            agent.target_network.load_state_dict(agent.q_network.state_dict())

        print(f"Episode {episode}: Total Reward = {total_reward}, Epsilon = {agent.epsilon:.3f}")

    return scores


# Main routine to initialize environment and start training
if __name__ == "__main__":

    mode = "train"  # Change to "test" for evaluation mode

    if mode == "test":
        env = gym.make("project2_env/RobotWorld-v0")
        agent.load("checkpoints/dqn_robot.pth", eval_mode=True)


    else:
        # Instantiate your custom environment
        # env = gym.make("project2_env/RobotWorld-v0", render_mode="human")
        env = gym.make("project2_env/RobotWorld-v0")

        # Define dimensions from the environment's observation and action spaces.
        state_dim = env.observation_space.shape[0]  # For example: 3 (x, y, theta)
        action_dim = env.action_space.n  # 22 discrete actions

        # Create a DQNAgent instance with chosen hyperparameters.
        agent = DQNAgent(state_dim, action_dim, hidden_dim=64, lr=1e-3,
                         gamma=0.99, epsilon_start=1.0, epsilon_end=0.05,
                         epsilon_decay=0.995, memory_capacity=10000, batch_size=64)

        # Train the agent
        scores = train_dqn(agent, env, num_episodes=200, target_update_interval=10)
        # Save the trained model
        agent.save("dqn_robotworld.pth")

        # Plot the rewards per episode
        plt.figure(figsize=(10, 5))
        plt.plot(scores, label='Episode Reward')
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Reward per Episode Over Training")
        plt.legend()
        plt.grid(True)
        plt.show()