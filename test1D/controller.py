from abc import ABC, abstractmethod
import numpy as np
from gymnasium import spaces
import os
import itertools
from minJerkHelperFunctions import submovement

from gym.spaces import Box

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
            f = 100
        elif(action == 1):
            f = -100
        # f = action
        extraCost = 0
        return f, extraCost
    
    def get_observation(self,env):
        # return np.array([env.x, env.x_dot])
        return np.array([env.x, env.x_dot,env.target],dtype=np.float32)


class controller_x0(controller):
    def __init__(self):
        super().__init__()

    def define_spaces(self):
        action_space = spaces.Box(-2,2,shape=(1,)) 

        # Obervation Space min and max
        target_range = [0.2, 0.5]
        position_range = [-1.0,2.0]
        velocity_range = [-20,20]

        observation_min = np.array([position_range[0],velocity_range[0],target_range[0]],dtype=np.float32)
        observation_max = np.array([position_range[1],velocity_range[1],target_range[1]],dtype=np.float32)

        observation_space = spaces.Box(low = observation_min, high=observation_max,shape=(3,)) 

        return action_space, observation_space

    def get_force(self, state, action, time):

        x = state[0]
        x_dot = state[1]

        x0 = action[0]
        x0_dot = 0 # action[1]

        zeta = 1
        wn = 50
        k = wn**2*self.m
        b = 2*zeta*wn*self.m
        
        f = k*(x0-x) + b*(x0_dot - x_dot)
        extraCost = 0

        return f, extraCost
    
    def get_observation(self,env):
        return np.array([env.x, env.x_dot,env.target],dtype=np.float32)

class controller_submovement(controller):
    def __init__(self):
        super().__init__()
        self.onGoingSubmovements = []
        self.completedSubmovementDiscplacement =[]
        self.initial_ib_b = 0
        self.thresholdLatency = 10
        self.latency = self.thresholdLatency + 1
        self.D_high = 0.05
        self.A_low = 0.01
        self.A_high = 0.2
        self.x0_completeSubmovements = 0

    def define_spaces(self):
        # Action Space Ranges min and max
        actionSelection_range = [0.0, 1.0]
        duration_range = [0.02, 0.2]
        amplitude_range = [0.1, 0.8]

        # Obervation Space min and max
        target_range = [0.2, 0.5]
        position_range = [-100,100]#[-1.0,2.0]
        velocity_range = [-1000,1000]#[-20,20]
        x0fhat_range = [-1000,1000]

        observation_min = np.array([position_range[0],velocity_range[0],target_range[0],x0fhat_range[0]],dtype=np.float32)
        observation_max = np.array([position_range[1],velocity_range[1],target_range[1],x0fhat_range[1]],dtype=np.float32)

        action_space = spaces.Discrete(5) # action can take values: 0, 1, 2, 3, 4
        observation_space = spaces.Box(low = observation_min, high=observation_max,shape=(4,)) 

        return action_space, observation_space
    
    def get_force(self, state, action, time):

        x = state[0]
        x_dot = state[1]

        x0,x0_dot,extraCost = self.getZFT(action, time)

        zeta = 1
        wn = 50
        k = wn**2*self.m
        b = 2*zeta*wn*self.m
        
        f = k*(x0-x) + b*(x0_dot - x_dot)

        return f, extraCost
    
    def get_observation(self,env):
        xfhat_0b_b, _ = self.sumOnGoingSubmovements_Vec(env.time + self.D_high) # Estiamte final position after final submovement ends
        xfhat_0b_b = np.float32(xfhat_0b_b)
        return np.array([env.x, env.x_dot,env.target,xfhat_0b_b],dtype=np.float32)
    
    def getZFT(self,action,time): # Later this will become get primatives
        # Because this function changes self.onGoingSubmovements only call it with step==True when using step function
        # in all other cases leave step == false

        # Import and scale all variables are originally 0-1
        # actionSelection =  True if action[0]>0.5 else False
        # duration = action[1]
        # amplitude = action[2]

        if(action == 0):
            actionSelection = False
        elif(action == 1):
            actionSelection = True
            duration = self.D_high
            amplitude = self.A_high
        elif(action == 2):
            actionSelection = True
            duration = self.D_high
            amplitude = self.A_low
        elif(action == 3):
            actionSelection = True
            duration = self.D_high
            amplitude = -self.A_high
        elif(action == 4):
            actionSelection = True
            duration = self.D_high
            amplitude = -self.A_low

        extraCost = 0
        
        # add to list
        if actionSelection: # Applied only when step == true
            self.add_submovement([duration, amplitude, time])
            extraCost = -10
            if (self.latency <= self.thresholdLatency):
                extraCost += -100
            self.latency = 0 # Reset latence after adding extra cost
        else:
            self.latency += 1

        # Remove submovements which are no longer active **ADD LATER TO MAKE MORE EFFICIENT
        # self.onGoingSubmovements = [submov for submov in self.onGoingSubmovements if (submov[0]+submov[3]) >= time]

        self.removeCompleteSubmovements(time)

        x_0b_b, x0_dot_b = self.sumOnGoingSubmovements_Vec(time)
        
        return x_0b_b, x0_dot_b, extraCost
    
    def add_submovement(self,sub):
        self.onGoingSubmovements.append(sub)
        # self.onGoingSubmovements.pop(0)
        # if len(self.onGoingSubmoements) > self.D_high 
        pass
    
    def sumOnGoingSubmovements(self,time):

        x0_tot = np.empty((len(self.onGoingSubmovements),))
        x0_dot_tot = np.empty((len(self.onGoingSubmovements),))

        # Sum up submovements
        for i in range(0,len(self.onGoingSubmovements)):            
            tmp1,tmp2 = submovement(self.onGoingSubmovements[i][0],
                                         self.onGoingSubmovements[i][1],
                                         self.onGoingSubmovements[i][2],time)
            
            x0_tot[i] = tmp1
            x0_dot_tot[i] = tmp2
            
        x_0i_b = np.sum(x0_tot)
        x0_dot_b = np.sum(x0_dot_tot)

        x_0b_b = x_0i_b + self.initial_ib_b

        return x_0b_b, x0_dot_b
    
    def sumOnGoingSubmovements_Vec(self,time):
                
        # print(result) 
        if(len(self.onGoingSubmovements) == 0):
            x0_tot = 0
            x0_dot_tot = 0
        else:
            sub = np.array(self.onGoingSubmovements)
            time_array = np.ones(sub.shape[0])*time
            sub_tuple = list(zip(sub[:,0],sub[:,1],sub[:,2],time_array))
            tmp = list(itertools.starmap(submovement,sub_tuple))
            x0_tot,x0_dot_tot = map(list,zip(*tmp))
            
        x_0i_b = np.sum(x0_tot)
        x0_dot_b = np.sum(x0_dot_tot)

        x_0b_b = x_0i_b + self.initial_ib_b + self.x0_completeSubmovements

        return x_0b_b, x0_dot_b
    
    def removeCompleteSubmovements(self,time):

        last_items = list(map(lambda x: x[-1], self.onGoingSubmovements))>time-self.D_high
        dex = list(itertools.compress(range(len(last_items)), last_items))
        
        # Add to standing x0
        if(len(dex) > 0):
            dex.sort(reverse=True) # Do in reverse order not to mess up indexing
            for i in dex:
                self.x0_completeSubmovements += self.onGoingSubmovements[i][0]
                self.onGoingSubmovements.pop(i)

        pass

    