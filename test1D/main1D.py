## Test with stable_baselines3
# Import Stable Baselines stuff
import os,time, sys
import numpy as np
from env1D import env1D
import gymnasium as gym
from modelClassical import modelClassical
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from typing import Callable
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from stable_baselines3.common.monitor import Monitor
from helperFunctions import defineDirectories

from sb3_contrib import RecurrentPPO

# tensorboard --logdir='Training/Logs'  

# from stable_baselines3.common.env_checker import check_env
# env = env1D('f')
# check_env(env)

# Ideas:
# - Recurrent nerual networks?
# - Liquid nerual networks

# http://localhost:6006/

# computationType = 'EvaluatePreLearning'
# computationType = 'Learn' 
# computationType = 'hardcode'
computationType = 'classical'
# computationType = 'hardcode - submovement'
# computationType = 'Evaluate'
# computationType = 'saveVideo'

saveName = 'test1D'

if( computationType == 'EvaluatePreLearning'):
# Look at it play untrained
    env = env1D(controllerType='x0',render_mode = 'human')
    obs = env.reset()
    score = 0
    terminated = 0
    episode = 0
    while not terminated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        print('Episode:{} Score:{} position{}, time:{}, action:{}'.format(episode, score, obs[0], env.time,action))
    env.close()

if( computationType == 'classical' ):
# Look at it play untrained
    env = env1D(controllerType='submovement',render_mode = 'human')
    obs,_ = env.reset()
    score = 0
    episode = 0
    count = 0
    terminated = False
    while not terminated:
        action = modelClassical(env,obs)
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        print('Episode:{} Score:{} position{}, time:{}, action:{}'.format(episode, score, obs[0], env.time,action))
    env.close()
    print('test')


if( computationType == 'hardcode'):

    # Add code to import a model
    controllerType = 'x0'

    # Hard code submovement as a sanity check
    env = env1D(controllerType,render_mode='human')
    obs = env.reset()
    done = False
    score = 0
    episode = 0
    count = 0
    N = 1.0/env.timeStep + 1
    x = np.zeros((int(N)))
    f = np.zeros((int(N)))
    while not done:
        # action = np.array([0.5], dtype=np.float32)
        action = model.predict(obs,deterministic=False)[0] # Use trained model
        obs, reward, terminated, truncated, info = env.step(action)
        # f[count],_ = env.robot.get_force(obs, action, env.time) # action[0]
        x[count] = obs[0]
        score += reward
        print('Episode:{} Score:{}'.format(episode, score))
        # if count >= 1:
        #     env.render()
        # count += 1
    env.close()
    timeVec = np.arange(N) * env.timeStep
    
    plt.figure()
    plt.plot(timeVec,x)
    plt.show()

    # plt.figure()
    # plt.plot(timeVec,f)
    # plt.show()
    print('test')

if( computationType == 'hardcode - submovement'):

    # Hard code submovement as a sanity check
    env = env1D()
    obs = env.reset()
    done = False
    score = 0
    episode = 0
    count = 0
    N = 1.0/env.timeStep + 1
    x = np.zeros((int(N)))
    while not done:
        if(count == 0):
            action = np.array([1.0,0.1,0.5]) # movement at time step 1
        else: 
            action = np.array([0.1,0.1,0.5]) # no movement other wise

        obs, reward, done, info = env.step(action)
        # f[count],_ = env.robot.get_force(obs, action, env.time) # action[0]
        x[count] = obs[0]
        score += reward
        print('Episode:{} Score:{}'.format(episode, score))
        if count >= 30:
            env.render(mode = 'human')
        count += 1
    env.close()
    timeVec = np.arange(N) * env.timeStep
    
    print('test')

if(computationType ==  'Learn'):
    
    controllerType = 'submovement'
    _, log_path, save_path, _ = defineDirectories(controllerType)  

    # Training
    env = env1D(controllerType)
    obs = env.reset()

    model = PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path)
    # model = RecurrentPPO("MlpLstmPolicy",env,verbose=1,tensorboard_log=log_path)

    # stop_callback = StopTrainingOnRewardThreshold(reward_threshold=700, verbose=1)
    # eval_callback = EvalCallback(env, 
    #                              callback_on_new_best=stop_callback, 
    #                              eval_freq=20_000, 
    #                              best_model_save_path=save_path, 
    #                              verbose=1)
    # model.learn(total_timesteps=1_000_000_000,callback=eval_callback)

    n_timesteps=50_000
    start_time = time.time()
    model.learn(total_timesteps=n_timesteps)
    total_time_multi = time.time() - start_time
    print(f"Took {total_time_multi:.2f}s for learn - {n_timesteps / total_time_multi:.2f} FPS")

    model.save(save_path)

    evaluate_policy(model, env, n_eval_episodes=10, render=None)