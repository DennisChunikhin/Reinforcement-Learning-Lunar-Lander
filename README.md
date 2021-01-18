# Reinforcement Learning Lunar Lander
Using reinforcement learning to train a computer how to land a lunar lander on uneven terrain.

## Deep Q Learning
I first tried implementing a Deep Q Network, however the network was unable to learn even the OpenAI gym LunarLander-v2 environment.

## Proximal Policy Optimization
I did some further research, and decided to implemented a proximal policy optimization network. This network was able to learn the LunarLander-v2 environment, so this is what I used further.

## Environment
For the environments, I modified the code for the LunarLander-v2 environment on OpenAI gym (the environment was created by Oleg Klimov)
I decided to test three cases

## No Terrain Data
No information about the terrain was included in the state given to the network.
The network was able to learn how to land.
![Training Graph](/imgs/LunarLanderTerrainNoData-v0.png)

## Height
The height of the lander above the terrain is included in the state.
The network was able to learn how to land much more smoothly than when no terrain data was given
![Training Graph](/imgs/LunarLanderTerrainHeight-v0.png)

## Terrain Data
The average slope of each chunk of terrain was included in the state.
The network took longer to learn and did not learn as much, however it was ultimately also successfull at landing.
![Training Graph](/imgs/LunarLanderTerrainData-v0.png)

## Conclusiong
I think that taking into account how long each network was trained for, the networks that recieved the height and the terrain data performed the best.
I think that this method could be scaled up to 3 dimensional terrain and greater numbers of chunks.

## Rough Terrain
I tweaked the environment further to create rougher terrain, and trained the network that recieved only the height to land.
The network was able to successfully learn how to land on the rough terrain.
![Training Graph](/imgs/LunarLanderRoughTerrain-v0.png)
