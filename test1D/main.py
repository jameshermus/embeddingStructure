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

# http://localhost:6006/

# computationType = 'EvaluatePreLearning'
computationType = 'Learn'
# computationType = 'Learn - Vectorized'
# computationType = 'hardcode'
# computationType = 'hardcode - submovement'
# computationType = 'Evaluate'

saveName = 'test1D'

if( computationType == 'EvaluatePreLearning' ):
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

if( computationType == 'hardcode'):

    # Add code to import a model
    saveName = 'PPO_x0_Learn.zip'
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
        action = model.predict(obs,deterministic=False)[0] # Use trained model
        obs, reward, done, info = env.step(action)
        f[count],_ = env.robot.get_force(obs, action, env.time) # action[0]
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

    plt.figure()
    plt.plot(timeVec,f)
    plt.show()
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
    
    saveName = 'PPO_x0_Learn'
    log_path = os.path.join('Training','Logs')
    save_path = os.path.join('Training','Saved_Models', saveName)

    # Training
    env = env1D()
    obs = env.reset()

    model = PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path)
    # stop_callback = StopTrainingOnRewardThreshold(reward_threshold=460, verbose=1)
    # eval_callback = EvalCallback(env, 
    #                              callback_on_new_best=stop_callback, 
    #                              eval_freq=20_000, 
    #                              best_model_save_path=save_path, 
    #                              verbose=1)
    # model.learn(total_timesteps=1_000_000_000,callback=eval_callback)
    model.learn(total_timesteps=1_000_000)

    model.save(save_path)

    evaluate_policy(model, env, n_eval_episodes=10, render=True)

if(computationType ==  'Learn - Vectorized'):

    saveName = 'PPO_x0_LearnVec'
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
    model = PPO('MlpPolicy',vec_env,verbose=1,tensorboard_log=log_path)
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=460, verbose=1)
    eval_callback = EvalCallback(vec_env, 
                                 callback_on_new_best=stop_callback, 
                                 eval_freq=20_000, 
                                 best_model_save_path=save_path, 
                                 verbose=1)
    model.learn(total_timesteps=100_000_000,callback=eval_callback)
    model.save(save_path)
    
    env = env1D()
    obs = env.reset()
    evaluate_policy(model, env, n_eval_episodes=10, render=True)

