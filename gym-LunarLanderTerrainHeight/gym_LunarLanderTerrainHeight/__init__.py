from gym.envs.registration import register

register(
    id='LunarLanderTerrainHeight-v1',
    entry_point='gym_LunarLanderTerrainHeight.envs:LunarLanderTerrainHeightEnv',
)