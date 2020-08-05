import random
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam

model_name = '24-48'
update_target_threshold = 5
replay_memory_size = 20000
minibatch_size = 32
episodes = 500
render_every = 50
lr = 0.001
gamma = 0.99

# for plotting
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}
average_every = 50

# min reward threshold to save
MIN_R = -100

class doubleDQN:
    def __init__(self, state_size, action_size, edecay):
        self.state_size = state_size
        self.action_size = action_size
        # model for training every minibatch
        self.model = self.create_model()

        # target model for predictions only
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.epsilon = 1
        self.epsilon_min = 0.001
        self.epsilon_decay = edecay

        # the replay memory prevents the training network from overfitting
        self.memory = deque(maxlen=replay_memory_size)

        self.target_update_counter = 0

    # default model used has 2 hidden layers (24-48)
    def create_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size,activation='relu'))
        model.add(Dense(48, activation='relu'))
        # linear activation for last layer because values are unbounded
        # for a regression problem
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',optimizer=Adam(lr=lr),metrics=['accuracy'])

        return model

    def replay_memory_append(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    # epsilon-greedy exploration/exploitation balance.
    def action(self,state):
        if np.random.random() > self.epsilon:
            # q = [[x,y]], q[0] = [x,y]
            action = np.argmax(self.model.predict(np.array(state).reshape(-1, *state.shape))[0])
        else:
            action = np.random.randint(0,env.action_space.n)
        
        return action
    
    def train_model(self, minibatch_size, final_state):
        # memory requires sufficient samples to create a minibatch
        if len(self.memory) < minibatch_size:
            return
        
        minibatch = random.sample(self.memory, minibatch_size)

        current_states = []
        next_states=[]

        # store current and next states in the minibatch into a list
        for state, action, reward, new_state, done in minibatch:
            current_states.append(state)
            next_states.append(new_state)

        current_states = np.array(current_states).reshape(minibatch_size, 2)
        next_states = np.array(next_states).reshape(minibatch_size, 2)

        # after creating a list of current and next states from within the minibatch, use .predict to 
        # generate 3 action values of current_q
        current_qs = self.model.predict(current_states)
        future_qs = self.target_model.predict(next_states)

        for index, (state, action, reward, new_state, done) in enumerate(minibatch):
            current_q = current_qs[index]
            if done:
                current_q[action] = reward
            else:
                max_future_q = max(future_qs[index])
                current_q[action] = reward + max_future_q * gamma
            
        self.model.fit(current_states, current_qs, epochs=1, verbose=0)
        
        if final_state:
            self.target_update_counter += 1
        
        # train target model when counter > (update_target_threshold)
        if self.target_update_counter > update_target_threshold:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        # epsilon decreases by a multiplicative factor of (epsilon_decay) every episode to limit the agent from
        # taking too many random actions
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# create mountaincar environment and initialize
env = gym.make('MountainCar-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
goal_position = env.goal_position

# initialize agent
agent = doubleDQN(state_size, action_size, edecay=0.995)

# begin episodic training
for episode in range(episodes):
    episode_reward = 0
    max_position = -99
    current_state = env.reset()

    done = False

    if episode % render_every == 0:
        render = True
    else:
        render = False

    while not done:
        # the default number of steps per episode is 200. The 'done' flag is asserted either when
        # the goal is reached or when 200 steps are completed.
        action = agent.action(current_state)

        new_state, reward, done, _ = env.step(action)

        episode_reward += reward

        if new_state[0] > max_position:
            max_position = new_state[0]

        if render:
            env.render()

        # create replay memory 
        agent.replay_memory_append(current_state,action,reward,new_state,done)

        agent.train_model(minibatch_size,done)

        if new_state[0] >= goal_position:
            print(f"target reached on episode: {episode}")
            break
        
        current_state = new_state

    print(f'episode: {episode} reward: {episode_reward}')

    # if the episode reward is above a minimum threshold, save that model
    if episode_reward >= MIN_R:
        agent.model.save(f'models/{model_name}-epreward{episode_reward}-maxpos{max_position}.h5')
        print(f'model saved for episode {episode}')

    ep_rewards.append(episode_reward)
    
    # aggregation plots
    if episode % average_every == 0:
        average_reward = sum(ep_rewards[-average_every:]) / len(ep_rewards[-average_every:])
        min_reward = min(ep_rewards[-average_every:])
        max_reward = max(ep_rewards[-average_every:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max_reward)
        aggr_ep_rewards['min'].append(min_reward)

plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['avg'],label='avg')
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['max'],label='max')
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['min'],label='min')
plt.legend()
plt.show()

    


            

