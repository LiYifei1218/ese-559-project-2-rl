import gymnasium as gym
import project2_env

# Initialise the environment
env = gym.make("project2_env/RobotWorld-v0", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset()
for _ in range(10000):
    # this is where you would insert your policy
    action = env.action_space.sample()

    print("Action taken:", action)

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.ret()

env.close()