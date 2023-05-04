## Test with stable_baselines3
# Import Stable Baselines stuff
import os,time, sys
from iiwaTest import iiwaTest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv 
# from stable_baselines3.common.vec_env import VecFrameStack # used to vectorize enviornment
from stable_baselines3.common.evaluation import evaluate_policy


# Training
env = iiwaTest(renderType = False)
# env = VecFrameStack(env, n_stack = 4) # A wrapper to tack the enviornment

log_path = os.path.join('Training','Logs')
model = PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path)

model.learn(total_timesteps=1000)

iiwaTest_path = os.path.join('Training','Saved_Models','iiwa_Model_PPO')
model.save(iiwaTest_path)

evaluate_policy(model,env, n_eval_episodes=10,render=False)


# # Look at it play untrained
# env_render = iiwaTest(renderType = True)
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

# # Evaluate saved Policty
# env_render = iiwaTest(renderType = True)
# log_path = os.path.join('Training','Logs')

# iiwaTest_path = os.path.join('Training','Saved_Models','iiwa_Model_PPO')

# model = PPO.load(iiwaTest_path)

# evaluate_policy(model,env_render, n_eval_episodes=10,render=False)