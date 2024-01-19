# Import GYM stuff
import gymnasium as gym
from scipy.spatial.transform import Rotation as R

# Import helpers
import os,time, sys
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import gymnasium as gym

from controllerMultiProc import controller, controller_f

class envMultiProc(gym.Env):
    def __init__(self,render_mode = None):
        super(envMultiProc, self).__init__()
        self.render_mode = render_mode

        self.time = 0
        self.fig = plt.figure()

        self.robot = controller_f()

        self.action_space, self.observation_space = self.robot.define_spaces()

        self.x = np.float32(0.0) #np.array([0.], dtype=np.float32)
        self.x_dot = np.float32(0.0)  # np.array([0.], dtype=np.float32)
        # self.target = np.array([0.5], dtype=np.float32)
        self.target = self.observation_space.sample()[2]
        
        self.timeStep = np.float32(1/750)
        self.timeMax = 0.75 #0.2
        self.tolerance_x = 0.05
        self.tolerance_x_dot = 0.002

        pass

    def reset(self, seed=None, options=None):

        super().reset(seed=seed, options=options)
        
        self.time = 0.0 # Reset time
        self.x = np.float32(0.0) #np.array([0.], dtype=np.float32)
        self.x_dot = np.float32(0.0) #np.array([0.], dtype=np.float32)
        self.target = self.observation_space.sample()[2]
        observation = self.robot.get_observation(self)

        if self.render_mode == "human":
            self._render_frame()

        info = {} # Empty dict
        
        return observation, info

    def step(self,action):

        # Step Dynamics (update self.x_dot, self.x, self.time)
        extraCost = self.stepDynamics(action)

        # Observation
        observation = self.robot.get_observation(self)
            
        # Let simulation run a fixed number of time steps
        if self.time >= self.timeMax:
            terminated = True
        else: 
            terminated = False

        if (abs(self.target - self.x) < self.tolerance_x):
            reward = 1
        else:
            reward = 0

        reward += extraCost

        truncated = False
            
        # Initialize info
        info = {}

        if self.render_mode == "human":
            self._render_frame()
     
        return observation, reward, terminated, truncated, info # Check may need to add truncated

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
        self.x_dot = self.x_dot + (np.float32(1)/self.robot.m) * f * self.timeStep
        self.x = self.x + self.x_dot * self.timeStep
        self.time += self.timeStep
        return extraCost