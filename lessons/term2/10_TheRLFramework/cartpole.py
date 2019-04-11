# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 22:19:19 2019

@author: shane
"""

import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()