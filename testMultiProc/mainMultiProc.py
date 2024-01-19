import numpy as np
import time

import gymnasium as gym
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from envMultiProc import envMultiProc
from stable_baselines3 import PPO, A2C


def main():
    # computationType = 'EvaluatePreLearning'
    # computationType = 'Learn' 
    computationType = 'Multiprocessing'

    saveName = 'testMultiProc'

    from stable_baselines3.common.env_checker import check_env
    env = envMultiProc()
    check_env(env, warn=True)

    if( computationType == 'EvaluatePreLearning'):
    # Look at it play untrained
        env = envMultiProc(render_mode = 'human')
        obs = env.reset()
        score = 0
        done = 0
        episode = 0
        while not done:
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            score += reward
            print('Episode:{} Score:{} position{}, time:{}, action:{}'.format(episode, score, obs[0], env.time,action))
        env.close()

    if(computationType ==  'Learn'):
        
        # Training
        env = envMultiProc()
        obs = env.reset()

        model = PPO('MlpPolicy',env,verbose=0)
        n_timesteps = 100_000

        start_time = time.time()
        model.learn(total_timesteps=n_timesteps)
        total_time = time.time() - start_time

        print(f"Took {total_time:.2f}s for learn - {n_timesteps / total_time:.2f} FPS")


    if(computationType ==  'Multiprocessing'):

        from typing import Callable


        def make_env(rank: int, seed: int = 0) -> Callable:
            """
            Utility function for multiprocessed env.

            :param env_id: (str) the environment ID
            :param num_env: (int) the number of environment you wish to have in subprocesses
            :param seed: (int) the inital seed for RNG
            :param rank: (int) index of the subprocess
            :return: (Callable)
            """

            def _init() -> gym.Env:
                env = envMultiProc()
                env.reset(seed=seed + rank)
                return env

            set_random_seed(seed)
            return _init
        
        num_cpu = 4  # Number of processes to use
        # Create the vectorized environment
        env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

        model = A2C("MlpPolicy", env, verbose=0)

        n_timesteps = 100_000
        start_time = time.time()
        model.learn(total_timesteps=n_timesteps)
        total_time = time.time() - start_time

        print(f"Took {total_time:.2f}s for learn - {n_timesteps / total_time:.2f} FPS")
        
        pass

if __name__ == '__main__':
    main()