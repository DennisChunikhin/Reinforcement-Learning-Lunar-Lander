from gym.envs.registration import register

register(
    id='LunarLanderTerrainData-v0',
    entry_point='gym_LunarLanderTerrainData.envs:LunarLanderTerrainDataEnv',
)