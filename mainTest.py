## Test with stable_baselines3
# Import Stable Baselines stuff
import os,time, sys
import numpy as np
from PyBulletRobotTest import PyBulletRobotTest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv 
from stable_baselines3.common.vec_env import VecFrameStack # used to vectorize enviornment
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt

# computationType = 'EvaluatePreLearning'
computationType = 'Learn'
# computationType = 'hardcode'
# computationType = 'Learn - Vectorized'
# computationType = 'Evaluate'

saveName = 'iiwa_tauControl'

if( computationType == 'EvaluatePreLearning'):
# Look at it play untrained
    env = PyBulletRobotTest(renderType = True)
    obs = env.reset()
    done = False
    score = 0
    episode = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        score += reward
        print('Episode:{} Score:{}'.format(episode, score))
    env.close()

if( computationType == 'hardcode'):
# Hard code submovement as a sanity check
    env = PyBulletRobotTest(renderType = True)
    obs = env.reset()
    done = False
    score = 0
    episode = 0
    count = 0
    N = 0.5/env.timeStep
    x0 = np.zeros((3,int(N)))
    x0_dot = np.zeros((3,int(N)))
    while not done:
        if( count == 100 ):
            # action = env.action_space.sample()
            action = np.array([[0.9],[0.2],[0.1],[np.pi*(0.0)]])# Apply submovement - Place selection greater than 0.5
        else:
            action = np.array([[0.1],[10.0],[10.0],[10.0]]) # No submovement - Place selection less than 0.5
        obs, reward, done, info = env.step(action)
        score += reward
        x_0b_b, x0_dot_b = env.robot.sumOnGoingSubmovements(env.time)
        x0[:,count] = x_0b_b.flatten()
        x0_dot[:,count] = x0_dot_b.flatten()
        print('Episode:{} Score:{}'.format(episode, score))
        count += 1
    env.close()

    print('test')
    print(env.robot.onGoingSubmovements)
    print(len(env.robot.onGoingSubmovements))
    timeVec = np.arange(N) * env.timeStep

    # X-Y position of robot
    fig, ax = plt.subplots(3)
    ax[0].plot(timeVec,x0[0,:])
    plt.ylabel('x')
    ax[1].plot(timeVec,x0[1,:])
    plt.ylabel('y')
    ax[2].plot(timeVec,x0[2,:])
    plt.ylabel('z')
    plt.xlabel('time')
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    plt.show()

    fig, ax = plt.subplots(3)
    ax[0].plot(timeVec,x0_dot[0,:])
    plt.ylabel('x_dot')
    ax[1].plot(timeVec,x0_dot[1,:])
    plt.ylabel('y_dot')
    ax[2].plot(timeVec,x0_dot[2,:])
    plt.ylabel('z_dot')
    plt.xlabel('time')
    plt.show()

if(computationType ==  'Learn'):

    # To do:
    # - Chagne to 0.5 seconds to speed up
    # - Try not changing target to make it faster at first
    # - check saving is working and that improviment is observed
    # - check tensorboard learn how modify
    # - add call backs
    # - Try to implament as an integration of velocity

    log_path = os.path.join('Training','Logs')
    iiwaTest_path = os.path.join('Training','Saved_Models', 'PPO_Submoement_2')

    # Training
    env = PyBulletRobotTest(renderType = False)
    obs = env.reset()

    model = PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path)

    # stop_callback = StopTrainingOnRewardThreshold(reward_threshold=-100, verbose=1)
    # eval_callback = EvalCallback(env, 
    #                              callback_on_new_best=stop_callback, 
    #                              eval_freq=10000, 
    #                              best_model_save_path=save_path, 
    #                              verbose=1)

    model.learn(total_timesteps=20_0000) #,callback=eval_callback)

    model.save(iiwaTest_path)

    evaluate_policy(model,env, n_eval_episodes=10,render=False)

if(computationType ==  'Learn - Vectorized'):

    # Create a function that returns your custom environment
    def make_env():
        return PyBulletRobotTest(renderType=False)

    # Number of parallel environments to run
    num_envs = 4

    # Create a list of environment functions
    env_fns = [make_env for _ in range(num_envs)]

    # Vectorize the environments
    vec_env = DummyVecEnv(env_fns)

    log_path = os.path.join('Training','Logs')
    model = PPO('MlpPolicy',vec_env,verbose=1,tensorboard_log=log_path)

    model.learn(total_timesteps=3000)

    iiwaTest_path = os.path.join('Training','Saved_Models', saveName)
    model.save(iiwaTest_path)

    evaluate_policy(model,vec_env, n_eval_episodes=10,render=False)

if ( computationType == 'Evaluate'):
    env_render = PyBulletRobotTest(renderType = False)
    log_path = os.path.join('Training','Logs')

    iiwaTest_path = os.path.join('Training','Saved_Models', saveName)

    model = PPO.load(iiwaTest_path)

    evaluate_policy(model,env_render, n_eval_episodes=10,render=False)


    # Look at it trained model
    episodes = 1
    xFinal = np.zeros((3,episodes))
    xStart = np.zeros((3,episodes))
    aFinal = np.zeros((3,episodes))
    distList = np.linspace(0.1,1,num=episodes)
    reward = np.zeros(episodes)
    actionAll = []

    for episode in range(1,episodes+1):
        obs = env_render.reset()
        xStart[0:3,episode-1] = obs[0:3,0]
        done = False
        score = 0
        while not done:
            # action = env_render.action_space.sample() # Sample action space
            action = model.predict(obs,deterministic=False)[0]            # Use trained model
            # action = np.array([0.3,distList[episode-1],0.3]) # Hard code
            obs, reward[episode-1], done, info = env_render.step(action)
        
        # aFinal[0:3,episode-1] = action
        xFinal[0:3,episode-1] = obs[0:3,0]
        # actionAll.append(action)
            
    env_render.close()

    bestIndex = np.argmax(reward)

    # X-Y position of robot
    target = np.array([[0.77982756],[0.2],[0.27314214]])
    fig, ax = plt.subplots()

    for i in range(0,episodes):
        # ax.scatter(xStart[i][0],xStart[i][1],xStart[i][2], marker='o',markersize=50)
        ax.plot(xFinal[0,i],xFinal[1,i],marker='o',markersize=10,color='r')
        ax.plot(xStart[0,i],xStart[1,i],marker='o',markersize=10,color='b')
        # print(i)
        # input('Press <ENTER> to continue')

    ax.plot(xFinal[0,bestIndex],xFinal[1,bestIndex],marker='o',markersize=10,color='y')
    ax.plot(target[0],target[1],marker='o',markersize=10,color='k')

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(reward)

    # Histogram of X,Y,Z
    names_xyz = ['x','y','z']
    fig, ax2 = plt.subplots(3)
    for i in range(0,3): 
        x = xFinal[i,:]
        counts, bins = np.histogram(x)
        ax2[i].stairs(counts,bins)
        countsMax = np.max(counts)
        ax2[i].plot([target[i],target[i]],[0,countsMax],color='k')
        ax2[i].set_xlim(0,1)
        ax2[i].set_title(names_xyz[i])
    plt.show()


    print('end of code')
