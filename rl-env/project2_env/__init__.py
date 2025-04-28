from gymnasium.envs.registration import register

register(
    id="project2_env/GridWorld-v0",
    entry_point="project2_env.envs:GridWorldEnv",
)

register(
    id="project2_env/RobotWorld-v0",
    entry_point="project2_env.envs:RobotWorldEnv",
)
register(
    id="project2_env/RobotWorldP3-v0",           # New unique ID for P3 env
    entry_point="project2_env.envs:RobotWorldEnvP3", # Points to the new class in the new file

)