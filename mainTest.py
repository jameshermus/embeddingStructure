## Test with stable_baselines3
# Import Stable Baselines stuff
import os,time, sys
import numpy as np
from PyBulletRobotTest import PyBulletRobotTest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv 
# from stable_baselines3.common.vec_env import VecFrameStack # used to vectorize enviornment
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt

# Look at it play untrained
env = PyBulletRobotTest(renderType = True)
obs = env.reset()
done = False
score = 0
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    score += reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()