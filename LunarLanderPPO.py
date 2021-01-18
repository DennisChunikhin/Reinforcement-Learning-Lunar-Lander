import random
import pylab
import numpy as np
import copy

import tensorflow as tf
from tensorboardX import SummaryWriter
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K

import gym
#Custom environments
import gym_LunarLanderTerrainNoData
import gym_LunarLanderTerrainHeight
import gym_LunarLanderTerrainData
import gym_LunarLanderRoughTerrain


class actor_model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        self.action_space = action_space

        X = Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
        X = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        output = Dense(self.action_space, activation="softmax")(X)

        self.actor = Model(inputs = X_input, outputs = output)
        self.actor.compile(loss=self.ppo_loss, optimizer=optimizer(lr=lr))
    
    #Loss function
    def ppo_loss(self, y_true, y_pred):
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001
        
        prob = actions * y_pred
        old_prob = actions * prediction_picks
        
        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)
        
        ratio = K.exp(K.log(prob) - K.log(old_prob))
        
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value = 1 - LOSS_CLIPPING, max_value = 1 + LOSS_CLIPPING) * advantages
        
        actor_loss = -K.mean(K.minimum(p1, p2))
        
        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)
        
        total_loss = actor_loss - entropy
        
        return total_loss
    
    def predict(self, state):
        return self.actor.predict(state)


class critic_model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        old_values = Input(shape=(1,))
        
        X = Dense(512, activation="relu", kernel_initializer="he_uniform")(X_input)
        X = Dense(256, activation="relu", kernel_initializer="he_uniform")(X)
        X = Dense(64, activation="relu", kernel_initializer="he_uniform")(X)
        value = Dense(1, activation=None)(X)
        
        self.critic = Model(inputs=[X_input, old_values], outputs = value)
        self.critic.compile(loss=[self.critic_PPO2_loss(old_values)], optimizer=optimizer(lr=lr))
    
    #Loss function
    def critic_PPO2_loss(self, values):
        def loss(y_true, y_pred):
            LOSS_CLIPPING = 0.2
            clipped_value_loss = values + K.clip(y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING)
            v_loss1 = (y_true - clipped_value_loss) ** 2
            v_loss2 = (y_true - y_pred) ** 2
            
            value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
            return value_loss
        return loss
    
    def predict(self, state):
        return self.critic.predict([state, np.zeros((state.shape[0], 1))])


class model:
    def __init__(self, env_name):
        # Initialization
        # Environment and PPO parameters
        self.env_name = env_name       
        self.env = gym.make(env_name)
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape
        self.EPISODES = 10000 # total episodes to train through all environments
        self.episode = 0 # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 0 # when average score is above 0 model will be saved
        self.lr = 0.00025
        self.epochs = 10 # training epochs
        self.shuffle=False
        self.training_batch = 1000
        self.optimizer = Adam

        self.replay_count = 0
        self.writer = SummaryWriter(comment="_"+self.env_name+"_"+self.optimizer.__name__+"_"+str(self.lr))
        
        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = [], [], [] # used in matplotlib plots

        # Create Actor-Critic network models
        self.actor = actor_model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        self.critic = critic_model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        
        self.name = f"{self.env_name}_PPO"
        
    def act(self, state):
        #predict the next action to take, using the model
        prediction = self.actor.predict(state)[0]
        action = np.random.choice(self.action_size, p=prediction)
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        return action, action_onehot, prediction
    
    #Generalized Advantage Estimation
    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.9, normalize = True):
        deltas = [r + gamma * (1-d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas)-1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t+1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)
    
    def replay(self, states, actions, rewards, predictions, dones, next_states):
        #Reshape memory to approproate shape for training
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        #Get Critic network predictions
        values = self.critic.predict(states)
        next_values = self.critic.predict(next_states)

        #Compute discounted rewards and advantages
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))

        #Stack everything to numpy array
        #Pack all advantags, predictions, and action to y_true and when they are received in custom PPO loss function we unpack it
        y_true = np.hstack([advantages, predictions, actions])

        #Training actor and critic networks
        a_loss = self.actor.actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=self.shuffle)
        c_loss = self.critic.critic.fit([states, values], target, epochs=self.epochs, verbose=0, shuffle=self.shuffle)
    
    #Save weights to file
    def save(self, name):
        self.actor.actor.save_weights(name+"_Actor.h5")
        self.critic.critic.save_weights(name+"_Critic.h5")
    
    #Load weights from file
    def load(self, actor_weights_name, critic_weights_name):
        self.actor.actor.load_weights(actor_weights_name)
        self.critic.critic.load_weights(critic_weights_name)
    
    #Save the model when it performs the best
    def save_best(self, score, episode):
        self.scores_.append(score)
        self.episodes_.append(episode)
        self.average_.append(sum(self.scores_[-50:]) / len(self.scores_[-50:]))
        
        #Save and plot every 100 episodes
        if str(episode)[-2:] == "00":
            self.save(self.name+"_Episode"+str(episode))
            
            pylab.plot(self.episodes_, self.scores_, 'b')
            pylab.plot(self.episodes_, self.average_, 'r')
            pylab.title(self.env_name+" PPO training cycle", fontsize=18)
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            try:
                pylab.grid(True)
                pylab.savefig(self.env_name+".png")
            except OSError:
                pass
        
        #Save best models
        if self.average_[-1] >= self.max_average:
            self.max_average = self.average_[-1]
            
            self.save(self.name)
            SAVING = "SAVING"
            
            #Learning rate decay
            self.lr *= 0.95
            K.set_value(self.actor.actor.optimizer.learning_rate, self.lr)
            K.set_value(self.critic.critic.optimizer.learning_rate, self.lr)
        else:
            SAVING = ""
        
        return self.average_[-1], SAVING
    
    #Train the model
    def run_batch(self):
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size[0]])
        done, score, SAVING, = False, 0, ''

        while True:
            #Initialize or reset memory
            states, next_states, actions, rewards, predictions, dones = [], [], [], [], [], []
            timesteps = 0
            for t in range(self.training_batch):
                timesteps += 1
                #self.env.render()
                action, action_onehot, prediction = self.act(state)
                next_state, reward, done, _ = self.env.step(action)

                states.append(state)
                next_states.append(np.reshape(next_state, [1, self.state_size[0]]))
                actions.append(action_onehot)
                rewards.append(reward)
                dones.append(done)
                predictions.append(prediction)

                state = np.reshape(next_state, [1, self.state_size[0]])
                score += reward

                if done or timesteps > 1000:
                    timesteps = 0
                    self.episode += 1
                    average, SAVING = self.save_best(score, self.episode)
                    print("episode: {}/{}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, score, average, SAVING))

                    state, done, score, SAVING = self.env.reset(), False, 0, ''
                    state = np.reshape(state, [1, self.state_size[0]])

            self.replay(states, actions, rewards, predictions, dones, next_states)
    
    #Run the model
    def run(self, num):
        for i in range(num):
            state = self.env.reset()
            state=np.reshape(state, [1, self.state_size[0]])
            done = False
            score = 0
            
            for t in range(1000):
                self.env.render()
                action = np.argmax(self.actor.predict(state)[0])
                state, reward, done, _ = self.env.step(action)
                state = np.reshape(state, [1, self.state_size[0]])
                score += reward

                if done:
                    break
            
            print("Score:", score)
                    
        self.env.close()


#TRAIN THE MODEL
env = "LunarLanderTerrainData-v0"
agent = model(env)
agent.run_batch()
