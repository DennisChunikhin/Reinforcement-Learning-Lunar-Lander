from gym.envs.registration import register

register(
    id='LunarLanderTerrainNoData-v0',
    entry_point='gym_LunarLanderTerrainNoData.envs:LunarLanderTerrainNoDataEnv',
)