import random
import numpy as np
from collections import deque
import os
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


import project2_env

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128): # Increased hidden_dim slightly
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
                 hidden_dim=128,        # Matched to QNetwork
                 lr=1e-4,               # Potentially smaller LR for stability
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.05,
                 epsilon_decay=0.9995,  # Slower decay for more exploration
                 memory_capacity=50000, # Increased buffer size
                 batch_size=64,
                 device=None):          # Allow specifying device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_capacity)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        print(f"Using device: {self.device}")

        # Initialize Q-network and target network
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # target network is used only for inference

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.training_steps = 0 # To track target network updates

    def select_action(self, state, training=True):
        # Use epsilon-greedy policy
        current_epsilon = self.epsilon if training else 0.0 # No exploration during eval
        if random.random() < current_epsilon:
            return random.randrange(self.action_dim)
        else:
            # Ensure state is numpy array before converting
            if not isinstance(state, np.ndarray):
                 state = np.array(state, dtype=np.float32)
            # Handle potential non-float types if necessary, though env should provide floats
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
         # Ensure states are stored as numpy arrays for consistency
        if not isinstance(state, np.ndarray): state = np.array(state, dtype=np.float32)
        if not isinstance(next_state, np.ndarray): next_state = np.array(next_state, dtype=np.float32)
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return 0 # Return 0 loss if not updating

        # Sample a random mini-batch of transitions
        transitions = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        # Convert to tensors
        # Use stack for efficiency and correct dimension handling
        states = torch.stack([torch.FloatTensor(s) for s in states]).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.stack([torch.FloatTensor(ns) for ns in next_states]).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)


        # Compute current Q values
        current_q = self.q_network(states).gather(1, actions)

        # Compute target Q values using Double DQN principle
        with torch.no_grad():
            # Select best action using the online network
            online_next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            # Evaluate the selected action using the target network
            max_next_q = self.target_network(next_states).gather(1, online_next_actions)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # Compute loss (Mean Squared Error)
        loss = nn.MSELoss()(current_q, target_q)

        # Perform a gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        #Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Decay epsilon for exploration
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        self.training_steps += 1
        return loss.item() # Return loss value

    # ---------- persistence ----------
    def save(self, path: str):
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "q_net":      self.q_network.state_dict(),
                "target_net": self.target_network.state_dict(),
                "optimizer":  self.optimizer.state_dict(),
                "epsilon":    self.epsilon,
                "training_steps": self.training_steps
            },
            path
        )
        print(f"[✓] Saved checkpoint to {path}")

    def load(self, path: str, eval_mode: bool = False):
        if not os.path.exists(path):
             print(f"[!] Checkpoint file not found at {path}")
             return False
        try:
            ckpt = torch.load(path, map_location=self.device)
            self.q_network.load_state_dict(ckpt["q_net"])
            self.target_network.load_state_dict(ckpt["target_net"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.epsilon = 0.0 if eval_mode else ckpt["epsilon"]
            self.training_steps = ckpt.get("training_steps", 0) # Load steps if available

            # Switch the policy net to eval() so layers  behave correctly
            if eval_mode:
                self.q_network.eval()
            else:
                self.q_network.train()
            self.target_network.eval() # Target network always in eval mode
            print(f"[✓] Loaded checkpoint from {path} (eval_mode={eval_mode}, epsilon={self.epsilon:.3f})")
            return True
        except Exception as e:
            print(f"[!] Failed to load checkpoint from {path}: {e}")
            return False


# --- Define Training Configurations ---
# Obstacles from Problem 2 description
obstacles_case1 = [
    {'x': -0.4, 'y': -0.4, 'r': 0.16},
    {'x': 0.1, 'y': -0.4, 'r': 0.16},
    {'x': -0.4, 'y': 0.1, 'r': 0.17}
]
obstacles_case2 = [
    {'x': -0.8, 'y': -0.4, 'r': 0.16},
    {'x': -0.1, 'y': -0.4, 'r': 0.17},
    {'x': 0.5, 'y': -0.4, 'r': 0.17}
]
obstacles_case3 = [
    {'x': -0.6, 'y': 1.0, 'r': 0.17},
    {'x': -0.6, 'y': -1.0, 'r': 0.16},
    {'x': 1.0, 'y': -0.6, 'r': 0.17}
]

# Convert obstacle dicts to numpy arrays expected by the (modified) env
def obstacles_to_array(obs_list):
    return np.array([[o['x'], o['y'], o['r']] for o in obs_list])

# Goals from Problem 3 description (x, y, tolerance)
goal_case1 = np.array([0.9, 0.9, 0.08])
goal_case2 = np.array([0.82, 0.95, 0.08])
goal_case3 = np.array([-1.0, 0.9, 0.08])

# Combine into configurations for training
# Each config dictionary should match what the modified RobotWorldEnv.reset expects
predefined_configs = [
    {"mode": "train3", "obstacles": obstacles_to_array(obstacles_case1), "goal": goal_case1},
    {"mode": "train3", "obstacles": obstacles_to_array(obstacles_case2), "goal": goal_case2},
    {"mode": "train3", "obstacles": obstacles_to_array(obstacles_case3), "goal": goal_case3},
]

# --- Function to Generate Random Configurations (Optional but Recommended) ---
def generate_random_config(num_obstacles=3, goal_tolerance=0.08):
    """Generates a random environment configuration for Problem 3 training."""
    # Generate random obstacles
    obstacles = []
    for _ in range(num_obstacles):
        # Simple generation, might need refinement to avoid impossible scenarios
        x = random.uniform(-1.1, 1.1) # Slightly smaller range to avoid edges
        y = random.uniform(-1.1, 1.1)
        r = random.uniform(0.16, 0.20)
        # Basic check to avoid obstacle overlap with typical start (-1.2, -1.2)
        if np.hypot(x - (-1.2), y - (-1.2)) > r + 0.1:
             obstacles.append({'x': x, 'y': y, 'r': r})
        else: # Retry if too close to start
             return generate_random_config(num_obstacles, goal_tolerance)

    if len(obstacles) < num_obstacles: # Fallback if generation failed often
         print("Warning: Could not generate enough valid obstacles, using Case 1 obstacles.")
         obstacles_arr = obstacles_to_array(obstacles_case1)
    else:
         obstacles_arr = obstacles_to_array(obstacles)


    # Generate random goal from one of the two ranges
    if random.random() < 0.5: # Range A: [0.8, 1] x [0.8, 1]
        goal_x = random.uniform(0.8, 1.0)
        goal_y = random.uniform(0.8, 1.0)
    else: # Range B: [-1.1, -0.9] x [0.8, 1]
        goal_x = random.uniform(-1.1, -0.9)
        goal_y = random.uniform(0.8, 1.0)

    goal_arr = np.array([goal_x, goal_y, goal_tolerance])

    # Basic check to avoid goal inside an obstacle
    for obs in obstacles_arr:
        if np.hypot(goal_arr[0] - obs[0], goal_arr[1] - obs[1]) < obs[2] + 0.05:
             # Goal is inside or too close to an obstacle, retry generation
             # print("Goal too close to obstacle, regenerating config...")
             return generate_random_config(num_obstacles, goal_tolerance)


    return {"mode": "train3", "obstacles": obstacles_arr, "goal": goal_arr}


# Training loop adapted for Problem 3
def train_dqn_p3(agent,
                 env_id="project2_env/RobotWorldP3-v0",
                 training_configs=None,
                 num_random_configs=20, # Number of additional random configs to generate
                 num_episodes=1000,     # Increased episodes
                 target_update_interval=5000, # Update target net based on steps, not episodes
                 max_steps_per_episode=300, # Limit episode length
                 save_interval=100,       # Save checkpoint every N episodes
                 save_path="checkpoints/dqn_p3_robot.pth",
                 render=False):
    """
    Trains the DQN agent for Problem 3 using multiple environment configurations.

    Args:
        agent: The DQNAgent instance.
        env_id: The Gymnasium environment ID.
        training_configs: A list of predefined configuration dictionaries.
        num_random_configs: How many random configurations to generate and add.
        num_episodes: Total number of training episodes.
        target_update_interval: Update target network every N training steps.
        max_steps_per_episode: Max steps before truncating an episode.
        save_interval: How often to save checkpoints (in episodes).
        save_path: Path to save the final model and checkpoints.
        render: Whether to render the environment during training (slow).
    """
    if training_configs is None:
        training_configs = predefined_configs

    # Generate and add random configurations
    print(f"Generating {num_random_configs} random training configurations...")
    for _ in range(num_random_configs):
        training_configs.append(generate_random_config())
    print(f"Total training configurations: {len(training_configs)}")

    scores = []
    avg_scores = [] # Moving average
    losses = []
    avg_losses = [] # Moving average

    # Create a single env instance - we'll reset it with different configs
    render_mode = "human" if render else None
    # We pass a dummy config initially, it will be overwritten by reset
    env = gym.make(env_id, render_mode=render_mode, config=training_configs[0])


    print(f"Starting training for {num_episodes} episodes...")
    for episode in range(1, num_episodes + 1):
        # Select a random configuration for this episode
        current_config = random.choice(training_configs)

        # Reset environment with the selected configuration
        # Pass the config dictionary directly to reset
        state, info = env.reset(options={'config': current_config})
        if not isinstance(state, np.ndarray): state = np.array(state, dtype=np.float32)


        total_reward = 0
        episode_loss = 0
        steps = 0
        done = False

        while not done and steps < max_steps_per_episode:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            if not isinstance(next_state, np.ndarray): next_state = np.array(next_state, dtype=np.float32)

            done = terminated or truncated # Environment handles termination/truncation

            # Store transition using numpy arrays
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.update() # Update returns the loss
            episode_loss += loss

            state = next_state
            total_reward += reward
            steps += 1

            # Update target network based on training steps
            if agent.training_steps % target_update_interval == 0:
                agent.target_network.load_state_dict(agent.q_network.state_dict())
                # print(f"--- Target network updated at step {agent.training_steps} ---")

            if render:
                env.render()


        scores.append(total_reward)
        avg_score = np.mean(scores[-100:]) # Moving average over last 100 episodes
        avg_scores.append(avg_score)

        step_loss = episode_loss / steps if steps > 0 else 0
        losses.append(step_loss)
        avg_loss = np.mean(losses[-100:])
        avg_losses.append(avg_loss)


        print(f"Ep {episode}/{num_episodes} | Steps: {steps} | R: {total_reward:.2f} | Avg R (100): {avg_score:.2f} | Avg Loss (100): {avg_loss:.4f} | Epsilon: {agent.epsilon:.3f}")

        # Save checkpoint periodically
        if episode % save_interval == 0:
            chkpt_path = save_path.replace(".pth", f"_ep{episode}.pth")
            agent.save(chkpt_path)

    env.close()
    print("Training finished.")
    # Save final model
    agent.save(save_path)
    return scores, avg_scores, losses, avg_losses


# Main routine
if __name__ == "__main__":
    # --- Configuration ---
    ENV_ID = "project2_env/RobotWorldP3-v0"
    NUM_EPISODES = 2000 # Adjust as needed
    TARGET_UPDATE_STEPS = 5000 # Adjust as needed
    SAVE_INTERVAL = 250 # Adjust as needed
    MODEL_SAVE_PATH = "checkpoints/p3_model.pth"
    LOAD_CHECKPOINT = None # Set to path like "checkpoints/p3_model_ep500.pth" to resume training
    RENDER_TRAINING = True # Set to True to watch training (very slow)

    # --- Environment Setup (for dimensions) ---
    # Need to instantiate once to get space dimensions
    temp_env = gym.make(ENV_ID)
    state_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.n
    temp_env.close()
    print(f"State Dim: {state_dim}, Action Dim: {action_dim}")

    # --- Agent Setup ---
    agent = DQNAgent(state_dim, action_dim,
                     hidden_dim=256,
                     lr=1e-4,
                     gamma=0.99,
                     epsilon_start=1.0,
                     epsilon_end=0.05,
                     epsilon_decay=0.9995, # Slower decay
                     memory_capacity=50000,
                     batch_size=64)

    # --- Load Checkpoint if specified ---
    if LOAD_CHECKPOINT:
        if agent.load(LOAD_CHECKPOINT, eval_mode=False):
             print(f"Resuming training from {LOAD_CHECKPOINT}")
        else:
             print(f"Could not load checkpoint {LOAD_CHECKPOINT}, starting fresh.")


    # --- Start Training ---
    scores, avg_scores, losses, avg_losses = train_dqn_p3(
        agent,
        env_id=ENV_ID,
        training_configs=predefined_configs, # Use predefined + generated random
        num_random_configs=30, # Generate 30 additional random scenarios
        num_episodes=NUM_EPISODES,
        target_update_interval=TARGET_UPDATE_STEPS,
        save_interval=SAVE_INTERVAL,
        save_path=MODEL_SAVE_PATH,
        render=RENDER_TRAINING
    )


    # --- Plotting Results ---
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:red'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Avg Return (100 episodes)', color=color)
    ax1.plot(avg_scores, color=color, label='Avg Return')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Avg Loss (100 episodes)', color=color)
    ax2.plot(avg_losses, color=color, linestyle='--', label='Avg Loss')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title("Problem 3 Training Progress (Avg Return & Loss)")
    # Add legend manually if needed, as twinx can be tricky
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.show()

    # Optional: Plot raw scores too
    plt.figure(figsize=(12, 5))
    plt.plot(scores, label='Raw Episode Reward')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Problem 3 Raw Reward per Episode")
    plt.legend()
    plt.grid(True)
    plt.show()
