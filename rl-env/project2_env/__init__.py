from gymnasium.envs.registration import register

register(
    id="project2_env/GridWorld-v0",
    entry_point="project2_env.envs:GridWorldEnv",
)
