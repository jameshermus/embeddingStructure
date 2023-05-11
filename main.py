## Test with stable_baselines3
# Import Stable Baselines stuff
import os,time, sys
import numpy as np
from PyBulletRobot import PyBulletRobot
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv 
# from stable_baselines3.common.vec_env import VecFrameStack # used to vectorize enviornment
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt

# # Look at it play untrained
# env_render = PyBulletRobot(renderType = True)
# episodes = 5
# for episode in range(1,episodes+1):
#     obs = env_render.reset()
#     done = False
#     score = 0
#     while not done:
#         action = env_render.action_space.sample()
#         obs, reward, done, info = env_render.step(action)
#         score += reward
#     print('Episode:{} Score:{}'.format(episode, score))
# env_render.close()

###########################
# # Training
# env = PyBulletRobot(renderType = False)
# # env = VecFrameStack(env, n_stack = 4) # A wrapper to tack the enviornment

# log_path = os.path.join('Training','Logs')
# model = PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path)

# model.learn(total_timesteps=1000)

# iiwaTest_path = os.path.join('Training','Saved_Models','iiwa_Model_PPO')
# model.save(iiwaTest_path)

# evaluate_policy(model,env, n_eval_episodes=10,render=False)

###########################
# Evaluate saved Policty
env_render = PyBulletRobot(renderType = False)
log_path = os.path.join('Training','Logs')

iiwaTest_path = os.path.join('Training','Saved_Models','iiwa_Model_PPO')

model = PPO.load(iiwaTest_path)

# evaluate_policy(model,env_render, n_eval_episodes=10,render=False)

# Look at it trained model
episodes = 100
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
    # while not done:
        
    action = env_render.action_space.sample() # Sample action space
    # action = model.predict(obs,deterministic=False)[0]            # Use trained model
    # action = np.array([0.3,distList[episode-1],0.3]) # Hard code
    obs, reward[episode-1], done, info = env_render.step(action)
    aFinal[0:3,episode-1] = action
    xFinal[0:3,episode-1] = obs[0:3,0]
    actionAll.append(action)
        
env_render.close()

bestIndex = np.argmax(reward)

# X-Y position of robot
target = np.array([[0.77982756],[0.2],[0.27314214]])
fig, ax = plt.subplots()
ax.plot(target[0],target[1],marker='o',markersize=10,color='k')

for i in range(0,episodes):
    # ax.scatter(xStart[i][0],xStart[i][1],xStart[i][2], marker='o',markersize=50)
    ax.plot(xFinal[0,i],xFinal[1,i],marker='o',markersize=10,color='r')
    ax.plot(xStart[0,i],xStart[1,i],marker='o',markersize=10,color='b')
    # print(i)
    # input('Press <ENTER> to continue')

ax.plot(xFinal[0,bestIndex],xFinal[1,bestIndex],marker='o',markersize=10,color='y')

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

# Histogram of duration, amplitude, direction
names_datheta = ['d','a','theta']
fig, ax3 = plt.subplots(3)
for i in range(0,3): 
    x = aFinal[i,:]
    counts, bins = np.histogram(x)
    ax3[i].stairs(counts,bins)
    countsMax = np.max(counts)
    # ax3[i].plot([aFinal[i,bestIndex],aFinal[i,bestIndex]],[0,countsMax],color='k') # Check shift
    ax3[i].set_xlim(0,1)
    ax3[i].set_title(names_datheta[i])
plt.show()

print('end of code')
