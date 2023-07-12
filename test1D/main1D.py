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
from typing import Callable
import matplotlib.pyplot as plt

from sb3_contrib import RecurrentPPO

# tensorboard --logdir='Training/Logs'  

from stable_baselines3.common.env_checker import check_env
env = env1D('x0')
check_env(env)

# Ideas:
# - Read about vectorize
# - Normalize?
# - Recurrent nerual networks?
# - Liquid nerual networks

# http://localhost:6006/

# computationType = 'EvaluatePreLearning'
# computationType = 'Learn' 
# computationType = 'Learn - Vectorized'
# computationType = 'hardcode'
# computationType = 'classical'
# computationType = 'hardcode - submovement'
# computationType = 'Evaluate'
computationType = 'Multiprocessing'

saveName = 'test1D'

if( computationType == 'EvaluatePreLearning'):
# Look at it play untrained
    env = env1D(render_mode = 'human')
    obs = env.reset()
    score = 0
    done = 0
    episode = 0
    while not done:
        action = env.action_space.sample()
        # action = np.array([100], dtype=np.float32)
        # action = np.array([0.5], dtype=np.float32)

        obs, reward, done, info = env.step(action)
        score += reward
        print('Episode:{} Score:{} position{}, time:{}, action:{}'.format(episode, score, obs[0], env.time,action))
    env.close()

if( computationType == 'classical'):
# Look at it play untrained
    env = env1D()
    obs = env.reset()
    score = 0
    done = 0
    episode = 0
    count = 0
    env.render(mode = 'human')
    while not done:
        action = modelClassical(env,obs)
        obs, reward, done, info = env.step(action)
        if count >= 1:
            env.render(mode = 'human')
        count += 1
        score += reward
        print('Episode:{} Score:{} position{}, time:{}, action:{}'.format(episode, score, obs[0], env.time,action))
    env.close()

if( computationType == 'hardcode'):

    # Add code to import a model
    saveName = 'PPO_f_Learn/best_model.zip'
    log_path = os.path.join('Training','Logs')
    save_path = os.path.join('Training','Saved_Models', saveName)
    model = PPO.load(save_path)

    # Hard code submovement as a sanity check
    env = env1D()
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
        # if count == 0:
        #     action = 2
        # elif count == 55:
        #     action = 2
        # else:
        #     action = 0
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
    
    saveName = 'PPO_f_Learn'
    log_path = os.path.join('Training','Logs')
    save_path = os.path.join('Training','Saved_Models', saveName)

    # Training
    env = env1D()
    obs = env.reset()

    model = PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path)
    # model = RecurrentPPO("MlpLstmPolicy",env,verbose=1,tensorboard_log=log_path)

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=700, verbose=1)
    eval_callback = EvalCallback(env, 
                                 callback_on_new_best=stop_callback, 
                                 eval_freq=20_000, 
                                 best_model_save_path=save_path, 
                                 verbose=1)
    model.learn(total_timesteps=1_000_000_000,callback=eval_callback)
    # model.learn(total_timesteps=50_000)

    model.save(save_path)

    evaluate_policy(model, env, n_eval_episodes=10, render=True)

if(computationType ==  'Learn - Vectorized'):

    saveName = 'PPO_f_LearnVec'
    log_path = os.path.join('Training','Logs')
    save_path = os.path.join('Training','Saved_Models', saveName)

    # Create a function that returns your custom environment
    def make_env():
        return env1D()

    # Number of parallel environments to run
    num_envs = 12
    # Create a list of environment functions
    env_fns = [make_env for _ in range(num_envs)]

    # Vectorize the environments
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecNormalize(vec_env)
    model = PPO('MlpPolicy',vec_env,verbose=1,tensorboard_log=log_path)
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=700, verbose=1)
    eval_callback = EvalCallback(vec_env, 
                                 callback_on_new_best=stop_callback, 
                                 eval_freq=20_000, 
                                 best_model_save_path=save_path, 
                                 verbose=1)
    model.learn(total_timesteps=100_000_000,callback=eval_callback)
    # model.learn(total_timesteps=1_000_000)
    model.save(save_path)
    
    env = env1D()
    obs = env.reset()
    evaluate_policy(model, env, n_eval_episodes=10, render=True)

    # terminal command: tensorboard --logdir={log_path}
    # tensorboard --logdir='Training/Logs/PPO_5/'

if(computationType ==  'Multiprocessing'):

    # Create a function that returns your custom environment
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
            env = env1D()
            env.reset(seed=seed + rank)
            return env

        set_random_seed(seed)
        return _init
    
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    env = DummyVecEnv([make_env(i) for i in range(num_cpu)])

    # env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    model = A2C("MlpPolicy", env, verbose=0)

    # By default, we use a DummyVecEnv as it is usually faster (cf doc)
    vec_env = make_vec_env(env_id, n_envs=num_cpu)

    model = A2C("MlpPolicy", vec_env, verbose=0)

    # We create a separate environment for evaluation
    eval_env = gym.make(env_id)

    # Random Agent, before training
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")

    n_timesteps = 25000

    # Multiprocessed RL Training
    start_time = time.time()
    model.learn(n_timesteps)
    total_time_multi = time.time() - start_time

    print(
        f"Took {total_time_multi:.2f}s for multiprocessed version - {n_timesteps / total_time_multi:.2f} FPS"
    )

    # Single Process RL Training
    single_process_model = A2C("MlpPolicy", env_id, verbose=0)

    start_time = time.time()
    single_process_model.learn(n_timesteps)
    total_time_single = time.time() - start_time

    print(
        f"Took {total_time_single:.2f}s for single process version - {n_timesteps / total_time_single:.2f} FPS"
    )

    print(
        "Multiprocessed training is {:.2f}x faster!".format(
            total_time_single / total_time_multi
        )
    )