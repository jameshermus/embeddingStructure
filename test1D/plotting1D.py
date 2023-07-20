import numpy as np
import matplotlib.pyplot as plt
import os
from env1D import env1D
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder
from helperFunctions import defineDirectories

class plotModels():
    def __init__(self,controllerType,dateInput = None):
        self.controllerType = controllerType
        self.dateInput = dateInput
        _, self.log_path, self.save_path, self.video_path = defineDirectories(controllerType,self.dateInput)  

        self.simulateModel()
        self.plotResults()
        pass
    
    def simulateModel(self):
        
        # Add code to import a model
        model = PPO.load(self.save_path)

        # Hard code submovement as a sanity check
        env = env1D(self.controllerType,render_mode=None)
        obs = env.reset()[0]
        terminated = False
        self.score = 0
        episode = 0
        count = 0
        self.N = int(1.0/env.timeStep)
        self.x = np.zeros(self.N)
        self.x_dot = np.zeros(self.N)
        self.f = np.zeros(self.N)
        self.target = obs[2]
        
        # Record the video starting at the first step
        # video_length = np.floor(750*0.75)
        # env = VecVideoRecorder(env, self.video_path,
        #                     record_video_trigger=lambda x: x == 0, video_length=video_length,
        #                     name_prefix=f"random-agent-{self.controllerType}")
        
        while not terminated:
            action,_ = model.predict(obs,deterministic=False) # Use trained model
            obs, reward, terminated, truncated, info = env.step(action)
            self.f[count],_ = env.robot.get_force(obs, action, env.time) 
            self.x[count] = obs[0]
            self.x_dot[count] = obs[1]
            # actionList.append(action.append(env.time))
            self.score += reward
            count += 1
            # print('Episode:{} Score:{}'.format(episode, self.score))
        env.close()
        self.timeVec = np.arange(self.N) * env.timeStep

        pass

    def plotResults(self):
        plt.figure()
        plt.plot(self.timeVec,self.x)
        plt.plot(self.timeVec,self.target*np.ones(self.N))
        plt.xlabel('Time(s)')
        plt.xlabel('x(m)')
        plt.show()
                
        plt.figure()
        plt.plot(self.timeVec,self.x_dot)
        plt.xlabel('Time(s)')
        plt.xlabel('v(m/s)')
        plt.show()

        plt.figure()
        plt.plot(self.timeVec,self.f)
        plt.xlabel('Time(s)')
        plt.xlabel('f(N)')
        plt.show()
        print(self.controllerType)
        pass

dateInput = '23-07-19'
# plotModels('f', dateInput) 
# plotModels('x0', dateInput)
plotModels('submovement', dateInput)