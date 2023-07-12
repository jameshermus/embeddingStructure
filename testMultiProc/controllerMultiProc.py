from abc import ABC, abstractmethod
import numpy as np
import os
from gymnasium import spaces


class controller(ABC):
    def __init__(self):
        self.m = np.float32(1) # Set mass (kg)
        pass
    
    @abstractmethod
    def define_spaces(self):
        # Each implamentation of the define_space method must define the action and observation spaces
        # This will very with diffrent types of x0 vs primative controls
        return action_space, obseration_space

    @abstractmethod
    def get_force(state,action,time):
        # Each implamentation must define the get_torque method. This method determins the torque to be produced
        # at each time step.
        return f, extraCost
    
    @abstractmethod
    def get_observation(self,env):

        return observation

class controller_f(controller):
    def __init__(self):
        super().__init__()

    def define_spaces(self):
        action_space = spaces.Discrete(2)

        # Obervation Space min and max
        target_range = [0.2, 0.5]
        position_range = [-1.0,2.0]
        velocity_range = [-20,20]

        observation_min = np.array([position_range[0],velocity_range[0],target_range[0]],dtype=np.float32)
        observation_max = np.array([position_range[1],velocity_range[1],target_range[1]],dtype=np.float32)

        observation_space = spaces.Box(low = observation_min, high=observation_max,shape=(3,)) 

        return action_space, observation_space

    def get_force(self, state, action, time):
        if(action == 0):
            f = np.float32(100)
        elif(action == 1):
            f = np.float32(-100)
        # f = action
        extraCost = 0
        return f, extraCost
    
    def get_observation(self,env):
        # return np.array([env.x, env.x_dot])
        return np.array([env.x, env.x_dot,env.target])
