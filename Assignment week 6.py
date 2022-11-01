#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import gym
import random
env = gym.make("Taxi-v3")
env.reset()
env.render()


# In[3]:


import numpy as np
import gym
import random

import pygame
pygame.init()
pygame.display.list_modes()

env = gym.make("Taxi-v3")
env.reset()
env.render()
env.action_space.n, env.observation_space.n
state = env.encode(4, 2, 3, 2) 
print("State:", state)
env.s = state
env.render()
num_episodes = 15000 #20000 #60000
gamma = 0.618 #0.99
learning_rate = 0.7 #0.95 #0.85
epsilon = 1

# initialize the Q table
Q = np.zeros([env.observation_space.n, env.action_space.n])
Q
def train_Qtable(Q, env, num_episodes, epsilon, gamma, lr_rate):
    '''
    function trains Q table with given parameters
    Args:
        Q (numpy array): Q table which will be updated
        env (gym environment)
        num_episodes (int): number of games that will be played during training
        epsilon (int): probability threshold
        gamma (int): discount rate
        lr_rate (int): learning rate
    Returns:
        Q_optimal (numpy array): updated Q table which is converged to optimal
    '''

    Q_old = Q.copy()
    for i in range(num_episodes):
        # define initial state
        state = env.reset()
        done = False
        while done == False:
            # First we select an action:
            if random.uniform(0, 1) < epsilon: # take a random number
                action = env.action_space.sample() # Explore action space
            else:
                action = np.argmax(Q[state,:]) # Exploit learned values
            # Then we perform the action and receive the feedback from the environment
            new_state, reward, done, info = env.step(action)
            # Finally we learn from the experience by updating the Q-value of the selected action
            update = reward + (gamma*np.max(Q[new_state,:])) - Q[state, action]
            Q[state,action] += learning_rate*update 
            if (Q_old == Q).all():
                print("Q table has been converged to optimal in {}th iteration ".format(i))
                return Q
            Q_old = Q.copy()
            state = new_state

    # even if Q table will not converge to optimal return latest updated Q table
    return Q
Q_optimal = train_Qtable(Q, env, num_episodes, epsilon, gamma, learning_rate)
print("Q table with optimal values:\n", Q_optimal )
def launch_game(Q, env):
    '''
    launch game with optimal Q value
    Args:
        Q (numpy array): Q table with optimal values
        env (gym environment)
    '''

    # define initial state
    state = env.reset()
    env.render()
    done = False
    while done == False:
        # Take the action (index) with the maximum expected discounted future reward given that state
        action = np.argmax(Q[state,:])
        state, reward, done, info = env.step(action)
        env.render()
print("\nlaunch game with optimal Q values\n")
launch_game(Q_optimal, env)


# In[ ]:





# In[ ]:





# In[ ]:




