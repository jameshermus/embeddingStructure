import numpy as np
import matplotlib.pyplot as plt
import os
from env1D import env1D
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from helperFunctions import defineDirectories

import imageio

import gym
import numpy as np
import cv2
from stable_baselines3 import PPO

class plotModels():
    def __init__(self,controllerType,dateInput = None):
        self.controllerType = controllerType
        self.dateInput = dateInput
        _, self.log_path, self.save_path = defineDirectories(controllerType,self.dateInput)  

        # Add code to import a model
        self.model = PPO.load(self.save_path+'/best_model')

        # Hard code submovement as a sanity check
        self.env = env1D(self.controllerType,render_mode='rgb_array')
        self.N = int(np.ceil(self.env.timeMax/self.env.timeStep))

        self.simulateModel()
        # self.record_video(video_length=self.N, video_file = self.save_path+"/video.mp4")
        self.plotResults()
        pass
    
    def simulateModel(self):
        
        obs = self.env.reset()[0]
        terminated = False
        self.score = 0
        episode = 0
        count = 0
        self.x = np.zeros(self.N)
        self.x_dot = np.zeros(self.N)
        self.f = np.zeros(self.N)
        self.target = obs[2]
        
        while not terminated:
            action,_ = self.model.predict(obs,deterministic=False) # Use trained model
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.f[count],_ = self.env.robot.get_force(obs, action, self.env.time) 
            self.x[count] = obs[0]
            self.x_dot[count] = obs[1]
            # actionList.append(action.append(env.time))
            self.score += reward
            count += 1
            # print('Episode:{} Score:{}'.format(episode, self.score))
        self.env.close()
        self.timeVec = np.arange(self.N) * self.env.timeStep

        pass


    # Function to record a video of the environment
    def record_video(self, video_length, video_file="video.mp4"):
        # Initialize video writer
        fps = int(1/self.env.timeStep)
        height, width, _ = self.env.render().shape
        video_writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        obs,_ = self.env.reset()
        for _ in range(video_length):
            # Render the environment
            frame = self.env.render()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Save frame to video
            video_writer.write(frame)

            # Take an action using the model
            action, _ = self.model.predict(obs)
            obs, _, truncated, _, _ = self.env.step(action)

            if truncated:
                obs, _ = self.env.reset()

        # Release video writer
        video_writer.release()


    def plotResults(self):
        plt.figure()
        plt.plot(self.timeVec,self.x)
        plt.plot(self.timeVec,self.target*np.ones(self.N),'k')
        plt.plot(self.timeVec,(self.target + self.env.tolerance_x)*np.ones(self.N),'--k')
        plt.plot(self.timeVec,(self.target - self.env.tolerance_x)*np.ones(self.N),'--k')
        plt.xlabel('Time(s)')
        plt.ylabel('x(m)')
        # plt.show()
        plt.savefig(self.save_path+"/position.png")
        plt.close(plt.gcf().number)
                
        plt.figure()
        plt.plot(self.timeVec,self.x_dot)
        plt.xlabel('Time(s)')
        plt.ylabel('v(m/s)')
        # plt.show()
        plt.savefig(self.save_path+"/velocity.png")
        plt.close(plt.gcf().number)

        plt.figure()
        plt.plot(self.timeVec,self.f)
        plt.xlabel('Time(s)')
        plt.ylabel('f(N)')
        # plt.show()
        plt.savefig(self.save_path+"/force.png")
        plt.close(plt.gcf().number)
        pass

dateInput = '23-07-24'
plotModels('f', dateInput) 
plotModels('x0', dateInput)
plotModels('submovement', dateInput)