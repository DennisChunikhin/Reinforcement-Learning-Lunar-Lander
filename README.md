# Reinforcement Learning Lunar Lander
Using reinforcement learning to train a computer how to land a lunar lander on uneven terrain.

##Deep Q Learning
I first tried implementing a Deep Q Network, however the network was unable to learn even the OpenAI gym LunarLander-v2 environment.

##Proximal Policy Optimization
I then implemented a proximal policy optimization network. This network was able to learn the environment, so this is what I used further.

##Environment
For the environments, I modified the code for the LunarLander-v2 environment on OpenAI gym (the environment was created by Oleg Klimov)
I decided to test three cases

##No Terrain Data
No information about the terrain was included in the state given to the network.
The network was able to learn how to land.

##Height
The height of the lander above the terrain is included in the state.
The network was able to learn how to land much more smoothly than when no terrain data was given

##Terrain Data
The average slope of each chunk of terrain was included in the state.
The network took longer to learn and did not learn as much, however it was ultimately also successfull at landing.

##Conclusiong
The network that was given only the height performed the best, however I think that given more tweaking and training, the network with terrain data could probably perform at least comparably.

##Rough Terrain
I tweaked the environment further to create rougher terrain, and trained the network that recieved only the height to land.
The network was able to successfully learn how to land on the rough terrain.
