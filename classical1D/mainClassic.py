## Test with stable_baselines3
# Import Stable Baselines stuff
import os,time, sys
import numpy as np
from env1D import env1D
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv 
from stable_baselines3.common.vec_env import VecFrameStack # used to vectorize enviornment
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import matplotlib.pyplot as plt

saveName = 'test1D'

# Look at it play untrained
env = env1D()
obs = env.reset()
score = 0
done = 0
episode = 0
count = 0
env.render(mode = 'human')
while not done:
    action = env.action_space.sample()
    # action = np.array([100], dtype=np.float32)
    # action = np.array([0.5], dtype=np.float32)

    obs, reward, done, info = env.step(action)
    if count >= 30:
        env.render(mode = 'human')
    count += 1
    score += reward
    print('Episode:{} Score:{} position{}, time{}'.format(episode, score, obs[0], env.time))
env.close()


