## Test with stable_baselines3
# Import Stable Baselines stuff
import os,time, sys
import numpy as np
from env1D import env1D
import gymnasium as gym
from modelClassical import modelClassical
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor

from typing import Callable
import matplotlib.pyplot as plt

from sb3_contrib import RecurrentPPO

# tensorboard --logdir='Training/Logs'  
# http://localhost:6006/

# Ideas:
# - Normalize?
# - Recurrent nerual networks?
# - Liquid nerual networks

def trainModel(controllerType,n_timesteps):

    log_path = os.path.join('Training','Logs',controllerType + str(n_timesteps))
    save_path = os.path.join('Training','Saved_Models', controllerType)

    def make_env(env_id: str,rank: int, seed: int = 0) -> Callable:
        """
        Utility function for multiprocessed env.

        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environment you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        :return: (Callable)
        """

        def _init() -> gym.Env:
            env = Monitor(env1D(env_id))
            env.reset(seed=seed + rank)
            return env

        set_random_seed(seed)
        return _init
    
    num_cpu = 20 #100  # Number of processes to use
    # Create the vectorized environment
    env = DummyVecEnv([make_env(controllerType,i) for i in range(num_cpu)])
    env = VecNormalize(env,norm_obs=True, norm_reward=True)

    model = PPO('MlpPolicy',env,verbose=0,tensorboard_log=log_path)
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=700, verbose=0)
    eval_callback = EvalCallback(env, 
                                 callback_on_new_best=stop_callback, 
                                 eval_freq=300_000, 
                                 best_model_save_path=save_path, 
                                 verbose=0)
    
    start_time = time.time()
    model.learn(total_timesteps=n_timesteps,callback=eval_callback)
    total_time = time.time() - start_time
    print(f"Controller Type", controllerType, f"took: {total_time:.2f}s")

    model.save(save_path)
    
    env_eval = env1D(controllerType,render_mode=None)
    obs = env_eval.reset()
    evaluate_policy(model, env_eval, n_eval_episodes=10, render=False)


n_timesteps = 10_000_000
controllerType = ['f','x0','submovement']
for i in range(len(controllerType)):
    trainModel(controllerType[i],n_timesteps)






