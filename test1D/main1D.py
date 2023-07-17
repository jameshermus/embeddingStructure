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


from sb3_contrib import RecurrentPPO

# tensorboard --logdir='Training/Logs'  

# from stable_baselines3.common.env_checker import check_env
# env = env1D('x0')
# check_env(env)

# Ideas:
# - Read about vectorize
# - Normalize?
# - Recurrent nerual networks?
# - Liquid nerual networks

# http://localhost:6006/

computationType = 'EvaluatePreLearning'
# computationType = 'Learn' 
# computationType = 'hardcode'
# computationType = 'classical'
# computationType = 'hardcode - submovement'
# computationType = 'Evaluate'
# computationType = 'saveVideo'

saveName = 'test1D'

if( computationType == 'EvaluatePreLearning'):
# Look at it play untrained
    env = env1D(controllerType='submovement',render_mode = 'human')
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


if( computationType == 'hardcode'):

    # Add code to import a model
    controllerType = 'x0'
    saveName = 'x02000000/best_model'
    log_path = os.path.join('Training','Logs')
    save_path = os.path.join('Training','Saved_Models', saveName)
    model = PPO.load(save_path)

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
    
    saveName = 'PPO_f_Learn'
    log_path = os.path.join('Training','Logs')
    save_path = os.path.join('Training','Saved_Models', saveName)

    # Training
    env = env1D('submovement')
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

if (computationType == 'saveVideo'):

    # export IMAGEIO_FFMPEG_EXE=/Applications/audio-orchestrator-ffmpeg/ffmpeg

    controllerType = 'f'

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
            env = Monitor(env1D(env_id,render_mode="rgb_array"))
            env.reset(seed=seed + rank)
            return env

        set_random_seed(seed)
        return _init
    

    video_folder = "videos/"
    video_length = 100
    num_proc = 4

    # Create the vectorized environment
    env = DummyVecEnv([make_env(controllerType,i) for i in range(num_proc)])
    env = VecNormalize(env,norm_obs=True, norm_reward=True)

    obs = env.reset()
    # Record the video starting at the first step
    # env = VecVideoRecorder(env, video_folder,
    #                     record_video_trigger=lambda x: x == 0, video_length=video_length,
    #                     name_prefix=f"random-agent-{controllerType}")

    env.reset()
    for _ in range(video_length + 1):
        action = [env.action_space.sample()]
        obs, _, _, _, _ = env.step(action)

    env.close() # Save video


