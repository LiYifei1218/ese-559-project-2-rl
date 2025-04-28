from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

import pickle

from project2 import move

class RobotWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, env_num=1):

        self.trace = []         # <-- add this

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

        self.env_num = env_num

        self.range_max = 1.0
        self.num_scans = 90
        self.scan_step = 0.05

        # 1) pose bounds
        pose_low  = [-1.3, -1.3, -1.0, -1.0]
        pose_high = [ 1.3,  1.3,  1.0,  1.0]

        # 2) raw goal‐position bounds (we know goal ∈ [−1.3,1.3]^2)
        goal_low  = [-1.3, -1.3]
        goal_high = [ 1.3,  1.3]

        # 3) goal‐vector bounds:
        #    dx,dy ∈ [−3,3],  d_goal ∈ [0, √18],  bearing ∈ [−π,π]
        gv_low  = [-3.0, -3.0, 0.0, -np.pi]
        gv_high = [ 3.0,  3.0, np.sqrt(18), np.pi]

        # 4) lidar scan bounds: distances ∈ [0, range_max]
        scan_low  = [0.0] * self.num_scans
        scan_high = [self.range_max] * self.num_scans

        low  = np.array(pose_low  + goal_low  + gv_low  + scan_low,
                        dtype=np.float32)
        high = np.array(pose_high + goal_high + gv_high + scan_high,
                        dtype=np.float32)

        self.observation_space = spaces.Box(low=low,
                                            high=high,
                                            dtype=np.float32)

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

        # goal
        goal = self._target_properties[:2]  # goal position

        # goal in robot frame
        dx, dy = goal - self._agent_state[:2]
        d_goal = np.hypot(dx, dy)
        bearing_err = np.arctan2(dy, dx) - th
        bearing_err = (bearing_err + np.pi) % (2 * np.pi) - np.pi  # wrap
        goal_vec = np.array([dx, dy, d_goal, bearing_err], dtype=np.float32)

        # # obstacle clearances
        # clearances = np.array(
        #     [np.linalg.norm([x - ox, y - oy]) - r
        #      for ox, oy, r in self._obstacles],
        #     dtype=np.float32
        # )

        # simulated "lidar" scan, 180 rays, 360° around robot
        scan = np.full(self.num_scans, self.range_max, dtype=np.float32)
        # precompute beam angles in world frame
        angles = np.linspace(0, 2*np.pi, self.num_scans, endpoint=False)
        for i, a in enumerate(angles):
            ray_angle = th + a
            # step outwards until we hit an obstacle or max range
            for d in np.arange(self.scan_step, self.range_max + 1e-6, self.scan_step):
                px = x + d * np.cos(ray_angle)
                py = y + d * np.sin(ray_angle)
                hit = False
                for ox, oy, orad in self._obstacles:
                    if (px - ox)**2 + (py - oy)**2 <= orad**2:
                        scan[i] = d
                        hit = True
                        break
                if hit:
                    break

        return np.concatenate((pose, goal, goal_vec, scan))

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

    def _generate_random_obstacles(self, num_obstacles=3):
        ob = np.zeros((num_obstacles, 3), dtype=np.float32)
        i = 0
        while i < num_obstacles:
            x = self.np_random.uniform(-1.3, 1.3)
            y = self.np_random.uniform(-1.3, 1.3)
            r = self.np_random.uniform(0.16, 0.20)
            # keep them away from start & goal
            if np.linalg.norm([x+1.2, y+1.2]) > 0.2 and \
               np.linalg.norm([x-1.2, y-1.2]) > 0.2:
                ob[i] = [x, y, r]
                i += 1
        return ob

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._elapsed_steps = 0

        mode = "test2"  # "test1", "test2", "train1", "train2"

        if mode == "train1":
            self._agent_state = np.array([-1.2, -1.2, 0.0])
            self._target_properties = np.array([1.2, 1.2, 0.08])
            self._obstacles = np.array([
                [-0.4, -0.4, 0.16],
                [0.1, -0.4, 0.16],
                [-0.4, 0.1, 0.17]
            ])


        elif mode == "test1":
            x_sample = np.random.uniform(-1.3, -1.2)
            y_sample = np.random.uniform(-1.3, -1.2)
            theta_sample = 0.0 #np.random.uniform(-np.pi, np.pi, size=(2,))

            self._agent_state = np.array([x_sample, y_sample, theta_sample])

            self._target_properties = np.array([1.2, 1.2, 0.08])

            self._obstacles = np.array([
                [-0.4, -0.4, 0.16],
                [0.1, -0.4, 0.16],
                [-0.4, 0.1, 0.17]
            ])

        elif mode == "train2":
            pass

        elif mode == "test2":
            self._agent_state = np.array([-1.2, -1.2, 0.0])
            self._target_properties = np.array([1.2, 1.2, 0.08])

            # randomly generate obstacles
            self._obstacles = self._generate_random_obstacles(num_obstacles=3)

        elif mode == "test3":
            self._agent_state = np.array([-1.2, -1.2, 0.0])
            self._target_properties = np.array([1.2, 1.2, 0.08])

            # randomly generate obstacles
            self._obstacles = self._generate_random_obstacles(num_obstacles=3)

        elif mode == "test9":
            x_sample = np.random.uniform(-1.3, 1.3)
            y_sample = np.random.uniform(-1.3, 1.3)
            theta_sample = np.random.uniform(-np.pi, np.pi)

            self._agent_state = np.array([x_sample, y_sample, theta_sample])

            x_sample = np.random.uniform(-1.3, 1.3)
            y_sample = np.random.uniform(-1.3, 1.3)

            self._target_properties = np.array([x_sample, y_sample, 0.08])

            # randomly generate obstacles
            self._obstacles = self._generate_random_obstacles(num_obstacles=3)


        # if test_case == 0:
        #     self._agent_state = np.array([-1.2, -1.2, 0.0])
        #
        # if test_case == 1:
        #
        #     x_sample = np.random.uniform(-1.3, -1.2)
        #     y_sample = np.random.uniform(-1.3, -1.2)
        #     theta_sample = 0.0 #np.random.uniform(-np.pi, np.pi, size=(2,))
        #
        #     self._agent_state = np.array([x_sample, y_sample, theta_sample])
        # elif test_case == 2:
        #     self._agent_state = np.array([-1.2, -1.2, 0.0])
        #
        # if self.env_num == 1:
        #     self._target_properties = np.array([1.2, 1.2, 0.08])
        #     self._obstacles = np.array([
        #         [-0.4, -0.4, 0.16],
        #         [0.1, -0.4, 0.16],
        #         [-0.4, 0.1, 0.17]
        #     ])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self.elapsed_steps = 0

        self.trace = []         # <-- clear trace at the start
        # record initial pos
        x, y, _ = self._agent_state
        self.trace.append((x, y))

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
        reward  = 10 * (prev_info["distance_to_goal"] - info["distance_to_goal"])
        reward += -0.5
        if info["min_obstacle_clear"] < 0.05:
            reward += -2.5 * (0.05 - info["min_obstacle_clear"])
        if info["collision"]:
            reward = -100
        elif info["goal_reached"]:
            reward = 100
        terminated = info["collision"] or info["goal_reached"]  or self._elapsed_steps > 300

        if self.render_mode == "human":
            self._render_frame()

        x, y, _ = self._agent_state
        self.trace.append((x, y))

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



        vis_robot_x = (self._agent_state[0] + 1.5) * vis_size
        vis_robot_y = (self._agent_state[1] + 1.5) * vis_size

        if len(self.trace) >= 2:
            pts = [((tx + 1.5) * vis_size, (ty + 1.5) * vis_size)
                   for tx, ty in self.trace]
            pygame.draw.lines(canvas,
                              (0, 0, 255),    # blue
                              False,          # not a closed loop
                              pts,
                              width=2)

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

        # First we draw the target
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (vis_target_x, vis_target_y),
            vis_size * self._target_properties[2],
        )

        # draw origin
        pygame.draw.circle(
            canvas,
            (0, 0, 0),
            (self.window_size / 2, self.window_size / 2),
            5
        )

        # # draw gridlines around origin (shifted by 1.5 units to center the grid)
        # for x in range(-1, 3):
        #     pygame.draw.line(
        #         canvas,
        #         0,
        #         (0, (x + 1.5) * vis_size),
        #         (self.window_size, (x + 1.5) * vis_size),
        #         width=3,
        #     )
        #     pygame.draw.line(
        #         canvas,
        #         0,
        #         ((x + 1.5) * vis_size, 0),
        #         ((x + 1.5) * vis_size, self.window_size),
        #         width=3,
        #     )
        #
        # # draw robot heading
        # pygame.draw.line(
        #     canvas,
        #     (0, 0, 255),
        #     (vis_robot_x, vis_robot_y),
        #     (
        #         vis_robot_x + vis_size * np.cos(self._agent_state[2]),
        #         vis_robot_y + vis_size * np.sin(self._agent_state[2]),
        #     ),
        #     width=3,
        # )

        # draw lidar scan
        for i, d in enumerate(self._get_obs()[6:]):
            angle = self._agent_state[2] + (i / self.num_scans) * 2 * np.pi
            px = vis_robot_x + d * vis_size * np.cos(angle)
            py = vis_robot_y + d * vis_size * np.sin(angle)
            pygame.draw.line(
                canvas,
                (0, 0, 255),
                (vis_robot_x, vis_robot_y),
                (px, py),
                width=1,
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
