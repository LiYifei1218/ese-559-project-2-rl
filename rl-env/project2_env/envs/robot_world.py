from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

import pickle

from project2 import move

class RobotWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        self.size = 3  # The size of the world
        self.window_size = 512  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).

        '''
        3 * 3 continuous world
        robot: x, y, theta
        N obstacles
        obstacle: x, y, radius
        goal: x, y, tolerance
        '''

        self._elapsed_steps = 0

        num_obstacles = 3

        # Robot state: [x, y, theta]
        robot_space = gym.spaces.Box(
            low=np.array([-1.3, -1.3, -np.pi], dtype=np.float32),
            high=np.array([1.3, 1.3, np.pi], dtype=np.float32),
            dtype=np.float32
        )

        # Obstacles: an array of N obstacles, each with [x, y, radius]
        obstacle_low = np.array([-1.5, -1.5, 0], dtype=np.float32)
        obstacle_high = np.array([1.5, 1.5, 3], dtype=np.float32)

        # Create a space with shape (num_obstacles, 3)
        obstacles_space = gym.spaces.Box(
            low=np.tile(obstacle_low, (num_obstacles, 1)),
            high=np.tile(obstacle_high, (num_obstacles, 1)),
            dtype=np.float32
        )

        # Goal: [x, y, tolerance]
        goal_space = gym.spaces.Box(
            low=np.array([-1.5, -1.5, 0], dtype=np.float32),
            high=np.array([1.5, 1.5, 3], dtype=np.float32),
            dtype=np.float32
        )

        # # Combine them into a Dict space
        # self.observation_space = gym.spaces.Dict({
        #     'agent': robot_space,
        #     'obstacles': obstacles_space,
        #     'goal': goal_space
        # })

        # self.observation_space = robot_space

        low = np.array([-1.3, -1.3, -1, -1,  # pose
                        -3, -3, 0, -np.pi,  # goal vec
                        -1.5, -1.5, -1.5  # clearances (≤0 means inside)
                        ], dtype=np.float32)
        high = np.array([1.3, 1.3, 1, 1,
                         3, 3, np.sqrt(18), np.pi,
                         1.5, 1.5, 1.5
                         ], dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # We have 22 actions
        self.action_space = spaces.Discrete(22)

        # Read action mappings from a pickle file
        with open("data/action_map_project_2.pkl", "rb") as f:
            action_mappings = pickle.load(f)

        self._action_to_direction = action_mappings

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    # def _get_obs(self):
    #     return self._agent_state
    #     # return {"agent": self._agent_state, "obstacles": self._obstacles, "goal": self._target_properties}

    def _get_obs(self):
        # pose
        x, y, th = self._agent_state
        pose = np.array([x, y, np.sin(th), np.cos(th)], dtype=np.float32)

        # goal in robot frame
        dx, dy = self._target_properties[:2] - self._agent_state[:2]
        d_goal = np.hypot(dx, dy)
        bearing_err = np.arctan2(dy, dx) - th
        bearing_err = (bearing_err + np.pi) % (2 * np.pi) - np.pi  # wrap
        goal_vec = np.array([dx, dy, d_goal, bearing_err], dtype=np.float32)

        # obstacle clearances
        clearances = np.array(
            [np.linalg.norm([x - ox, y - oy]) - r
             for ox, oy, r in self._obstacles],
            dtype=np.float32
        )

        return np.concatenate((pose, goal_vec, clearances))

    def _get_info(self):
        d_goal = np.linalg.norm(self._agent_state[:2] - self._target_properties[:2])
        min_clearance = min(
            np.linalg.norm(self._agent_state[:2] - obs[:2]) - obs[2]
            for obs in self._obstacles
        )
        return {
            "distance_to_goal": d_goal,
            "min_obstacle_clear": min_clearance,
            "goal_reached": d_goal < self._target_properties[2],
            "collision": min_clearance < 0,
            "step": self._elapsed_steps  # if you track it
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._elapsed_steps = 0

        env_num = 1

        if env_num == 1:
            self._agent_state = np.array([-1.2, -1.2, 0.0])
            self._target_properties = np.array([1.2, 1.2, 0.08])
            self._obstacles = np.array([
                [-0.4, -0.4, 0.16],
                [0.1, -0.4, 0.16],
                [-0.4, 0.1, 0.17]
            ])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self.elapsed_steps = 0

        return observation, info

    def step(self, action):

        prev_info = self._get_info()

        linear_velocity = self._action_to_direction[action][0]
        angular_velocity = self._action_to_direction[action][1]

        # call move function in project pkg to update the agent's state
        self._agent_state = move(
            self._agent_state,
            (linear_velocity, angular_velocity)
        )

        # # An episode is done if the agent has reached the target
        # goal_reached = np.linalg.norm(
        #     self._agent_state[:2] - self._target_properties[:2]
        # ) < self._target_properties[2]
        #
        # # or collision with obstacles
        # collision = False
        # for obs in self._obstacles:
        #     if np.linalg.norm(self._agent_state[:2] - obs[:2]) < obs[2]:
        #         collision = True
        #         break
        #
        # terminated = goal_reached or collision

        self._elapsed_steps += 1
        obs, info = self._get_obs(), self._get_info()

        # ------- reward shaping -------
        reward  = 3 * (prev_info["distance_to_goal"] - info["distance_to_goal"])
        reward += -0.1
        if info["min_obstacle_clear"] < 0.05:
            reward += -10 * (0.05 - info["min_obstacle_clear"])
        if info["collision"]:
            reward = -100
        elif info["goal_reached"]:
            reward = 100
        terminated = info["collision"] or info["goal_reached"]  or self._elapsed_steps > 200

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        vis_size = (
            self.window_size / self.size
        )  # The size of a unit square in pixels

        vis_target_x = (self._target_properties[0] + 1.5) * vis_size
        vis_target_y = (self._target_properties[1] + 1.5) * vis_size

        # First we draw the target
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (vis_target_x, vis_target_y),
            vis_size * self._target_properties[2],
        )

        vis_robot_x = (self._agent_state[0] + 1.5) * vis_size
        vis_robot_y = (self._agent_state[1] + 1.5) * vis_size

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (vis_robot_x, vis_robot_y),
            vis_size / 16,
        )

        # Finally, add obstacles
        for obs in self._obstacles:
            vis_obstacle_x = (obs[0] + 1.5) * vis_size
            vis_obstacle_y = (obs[1] + 1.5) * vis_size
            pygame.draw.circle(
                canvas,
                (0, 255, 0),
                (vis_obstacle_x, vis_obstacle_y),
                vis_size * obs[2],
            )

        # draw origin
        pygame.draw.circle(
            canvas,
            (0, 0, 0),
            (self.window_size / 2, self.window_size / 2),
            5
        )

        # draw gridlines around origin (shifted by 1.5 units to center the grid)
        for x in range(-1, 3):
            pygame.draw.line(
                canvas,
                0,
                (0, (x + 1.5) * vis_size),
                (self.window_size, (x + 1.5) * vis_size),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                ((x + 1.5) * vis_size, 0),
                ((x + 1.5) * vis_size, self.window_size),
                width=3,
            )


        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
