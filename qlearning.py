import gym
import numpy as np
import matplotlib.pyplot as plt

lr = 0.01
discount = 0.99
episodes = 25000
render_every = 500
epsilon = 0.1

# episode range in which epsilon decay occurs
start_epsilon_decaying = 1
end_epsilon_decaying = episodes // 2
epsilon_decay_value = epsilon / (end_epsilon_decaying - start_epsilon_decaying)
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

env = gym.make('MountainCar-v0')
env.reset()
# each state is a combination of [position, velocity]
# discretize the states
discrete_obs_size = 20
discrete_window_size = (env.observation_space.high - env.observation_space.low) / discrete_obs_size

# 20x20x3 q table, 20x20 represents the discretized combinations of [position,velocity] available, 3 is the number of actions possible
q_table = np.random.uniform(low=-2, high=0,size=(discrete_obs_size,discrete_obs_size,env.action_space.n))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_window_size
    return tuple(discrete_state.astype(np.int))

for episode in range(episodes):
    episode_reward = 0

    # only renders every (render_every) episodes
    if episode % render_every == 0:
        render = True
    else:
        render = False
    discrete_state = get_discrete_state(env.reset())
    done = False

    while not done:
        if np.random.random() > epsilon: # greedy
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0,env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        

        new_discrete_state = get_discrete_state(new_state)
        if render:
            #print('rendering...')
            env.render()
        if not done:
            max_q = np.max(q_table[new_discrete_state]) # maximum q based on next state
            current_q = q_table[discrete_state + (action,)]
            # q learning update rule
            new_q = (1 - lr)*current_q + lr*(reward + discount*max_q) 

            # we have taken the step already and gotten the reward, but based on that step we are 
            # updating that action we just took with a new q-value
            q_table[discrete_state + (action,)] = new_q

        # if the position entry of the new state is larger than the goal position, then set the 
        # q value of that action to 0    
        elif new_state[0] >= env.goal_position:
            print(f"done on episode: {episode}")
            print(done)
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

    if start_epsilon_decaying <= episode <= end_epsilon_decaying:
        epsilon -= epsilon_decay_value

    print(f'episode: {episode} reward: {episode_reward}')
    ep_rewards.append(episode_reward)
    if episode % render_every == 0:
        # average reward takes the total rewards of each episode for the last (render_every) episodes
        # and performs an average
        average_reward = sum(ep_rewards[-render_every:]) / len(ep_rewards[-render_every:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(ep_rewards[-render_every:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-render_every:]))


env.close()

plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['avg'],label='avg')
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['max'],label='max')
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['min'],label='min')
plt.legend()
plt.show()

