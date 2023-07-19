# Import GYM stuff
import gymnasium as gym
from scipy.spatial.transform import Rotation as R

# Import helpers
import os,time, sys
import numpy as np
import random
import math
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from controller import controller, controller_f, controller_x0, controller_submovement

class env1D(gym.Env):
    def __init__(self,controllerType,render_mode = None):
        super(env1D, self).__init__()
        
        self.render_mode = render_mode
        self.controllerType = controllerType
        
        if self.render_mode == 'human':
            self.fig = plt.figure()
        else:
            self.fig = []

        self.time = np.float32(0.0)

        # Add panda, table, box, and object
        if(controllerType == 'f'):
            self.robot = controller_f()
        elif(controllerType == 'x0'):
            self.robot = controller_x0()
        elif(controllerType == 'submovement'):
            self.robot = controller_submovement()
        else:
            raise Exception('Invalid controller type')


        self.action_space, self.observation_space = self.robot.define_spaces()

        self.x = np.float32(0.0)
        self.x_dot = np.float32(0.0)
        self.target = self.observation_space.sample()[2]
        
        self.timeStep = np.float32(1/750)
        self.timeMax = 0.75
        self.tolerance_x = 0.05
        self.tolerance_x_dot = 0.002

        pass


    def reset(self, seed=None, options=None):

        super().reset(seed=seed, options=options)
        
        self.time = np.float32(0) # Reset time
        self.x = np.float32(0.0)
        self.x_dot = np.float32(0.0)
        self.target = self.observation_space.sample()[2]
        observation = self.robot.get_observation(self)

        if self.render_mode == "human":
            self._render_frame()

        info = {} # Empty dict
        
        return observation, info

    def step(self,action):

        # start_time = time.time()

        # self.prev_x_dot = self.x_dot

        # Step Dynamics (update self.x_dot, self.x, self.time)
        extraCost = self.stepDynamics(action)

        actionNull = 0
        downSampleFactor = 15
        if self.controllerType == 'submovement':
            for i in range(downSampleFactor):
                self.stepDynamics(actionNull)
     
        # Observation
        observation = self.robot.get_observation(self)
            
        # Let simulation run a fixed number of time steps
        if self.time >= self.timeMax:
            terminated = True
        else: 
            terminated = False

        truncated = False

        # Define the target position and sigma
        # sigma = 0.075 

        # Evaluate the probability density function at each x position
        # reward = norm.pdf(self.x, loc=self.target, scale=sigma)[0]  
        # reward = - np.linalg.norm(self.x-self.target)**2  - 0.001*np.linalg.norm(self.prev_x_dot-self.x_dot)**2
        # - 0.1*np.linalg.norm(self.x_dot)**2

        if (abs(self.target - self.x) < self.tolerance_x): #& (abs(self.x_dot) < tolerance_x_dot):
            reward = 1
        else:
            reward = 0

        reward += extraCost

        # tmp = np.zeros((100,))
        # for i in range(-1,2,100):
        #     tmp[i] = norm.pdf(i, loc=self.target, scale=sigma)[0]**2

        # error = self.target-self.x
        # reward = -1*np.matmul(error.transpose(),error) + extraCost # distance cost per time step
        
        # reward = -0.1

        # # Assign Reward
        # delta_x = 0.1
        # delta_x_dot = 0.01
        # if ( (self.target-delta_x < self.x) and (self.x < self.target+delta_x) and (np.abs(self.x_dot) < delta_x_dot)):
        #     reward += 10000
        #     done = True
        # elif   ( (self.target-delta_x < self.x) and (self.x < self.target+delta_x)):
        #     reward += 10
            # done = True
        # else: 
        #     reward = 0
        #     # done = False
            
        # Initialize info
        info = {}

        if self.render_mode == "human":
            self._render_frame()

        # total_time_multi = time.time() - start_time
        # print(f"Took {total_time_multi:.8f}s for learn")
     
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'human':
            return self._render_frame()
    
    def _render_frame(self):
        # X-Y position of robot
        self.fig.clear()
        plt.plot(self.x,0,'k.',markersize=30)
        plt.plot([0.0,0.0],[-0.1,0.1],'k')
        plt.plot([self.target-self.tolerance_x,self.target-self.tolerance_x],[-0.1,0.1],'k')
        plt.plot([self.target+self.tolerance_x,self.target+self.tolerance_x],[-0.1,0.1],'k')
        plt.title(format(self.time, ".2f"))
        # plt.ylabel('x')
        # plt.ylabel('y')
        # plt.ylim([-0.5, 0.5])
        plt.xlim([-0.1,1.0])
        plt.pause(0.0001)
        # plt.show()
    
    def close(self):
        pass

    def stepDynamics(self,action):   
        state = self.robot.get_observation(self)
        f, extraCost = self.robot.get_force(state, action, self.time)
        self.x_dot = self.x_dot + (1/self.robot.m) * f * self.timeStep
        self.x = self.x + self.x_dot * self.timeStep
        self.time += self.timeStep
        return extraCost