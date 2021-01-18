from gym.envs.registration import register

register(
    id='LunarLanderRoughTerrain-v0',
    entry_point='gym_LunarLanderRoughTerrain.envs:LunarLanderRoughTerrainEnv',
)