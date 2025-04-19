import argparse
import numpy as np
import torch
import gymnasium as gym

import project2_env
from p1          import DQNAgent               # the class you used to train


# --------------------------------------------------------------------------- #
#                     ----- 1.  argument parsing -----                        #
# --------------------------------------------------------------------------- #
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",     required=True,
                        help="Path to the .pth checkpoint file")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of evaluation episodes")
    parser.add_argument("--device",   default="cpu",
                        choices=["cpu", "cuda", "mps"],
                        help="Torch device for inference")
    parser.add_argument("--render",   action="store_true",
                        help="Call env.render() each step (slow)")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
#                          ----- 2.  make env -----                           #
# --------------------------------------------------------------------------- #
def make_test_env():
    """
    Create a *different* map than training.
    Replace the obstacles list below with whatever scenario you want to test.
    """
    # test_obstacles = [(-0.2, -0.5, 0.14),      # (x, y, r)
    #                   ( 0.3, -0.3, 0.18),
    #                   (-0.1,  0.3, 0.15)]
    env = gym.make("project2_env/RobotWorld-v0", render_mode="human")
    return env


# --------------------------------------------------------------------------- #
#                       ----- 3. evaluation loop -----                        #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(agent, env, n_episodes=50, render=False):
    returns, steps, successes = [], [], []
    for ep in range(n_episodes):



        obs, _ = env.reset()
        total_r, n = 0.0, 0
        done = False

        while not done:
            # get the state from the observation
            # pose = obs[:4]  # [x, y, sin(th), cos(th)]

            action = agent.select_action(obs, training=False)

            obs, reward, done, truncated, info = env.step(action)
            total_r += reward
            n += 1
            if render:
                env.render()

        returns.append(total_r)
        steps.append(n)
        successes.append(info.get("goal_reached", False))

    return (np.array(returns), np.array(steps), np.array(successes))


# --------------------------------------------------------------------------- #
#                           ----- 4.  main  -----                             #
# --------------------------------------------------------------------------- #
def main():
    args = get_args()
    env = make_test_env()

    # Re‑create agent skeleton (same net dims as training)
    agent = DQNAgent(
        state_dim  = env.observation_space.shape[0],
        action_dim = env.action_space.n,
        hidden_dim = 128,                 # keep identical to training run
    )
    agent.load(args.ckpt, eval_mode=True)      # sets ε = 0 and .eval()

    returns, steps, successes = evaluate(agent, env,
                                         n_episodes=args.episodes,
                                         render=args.render)

    # ---------------- results ----------------
    print("\nEvaluation over {} episodes".format(args.episodes))
    print("  Success rate     : {:.1f}%"
          .format(100 * successes.mean()))
    print("  Mean episode len : {:.1f} ± {:.1f} steps"
          .format(steps.mean(), steps.std()))
    print("  Mean return      : {:.1f} ± {:.1f}"
          .format(returns.mean(), returns.std()))
    print("  Min / Max return : {:.1f} / {:.1f}"
          .format(returns.min(),  returns.max()))


if __name__ == "__main__":
    main()