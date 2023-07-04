from abc import ABC, abstractmethod
import numpy as np
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
import os

from gym.spaces import Box

class controller(ABC):
    def __init__(self):
        self.m = 1 # Set mass (kg)
        pass
    
    @abstractmethod
    def define_spaces(self):
        # Each implamentation of the define_space method must define the action and observation spaces
        # This will very with diffrent types of x0 vs primative controls
        return action_space, obseration_space

    @abstractmethod
    def get_force(state, action,time):
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
        action_space = Discrete(2)
        # action_space = Box(-100,100,shape=(1,))         
        observation_space = Box(low = np.array([[-1],[2]],dtype=np.float32),
                               high = np.array([[-4000],[4000]],dtype=np.float32),shape=(2,1)) 
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
        return np.array([env.x, env.x_dot])

class controller_x0(controller):
    def __init__(self):
        super().__init__()

    def define_spaces(self):
        action_space = Box(-1,2,shape=(1,)) 
        observation_space = Box(low = np.array([[-1],[2]],dtype=np.float32),
                                high = np.array([[-20],[20]],dtype=np.float32),shape=(2,1))

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
        return np.array([env.x, env.x_dot])

class controller_submovement(controller):
    def __init__(self):
        super().__init__()
        self.onGoingSubmovements = []
        self.initial_ib_b = 0

    def define_spaces(self):
        # Action Space Ranges min and max
        actionSelection_range = [0.0, 1.0]
        duration_range = [0.02, 0.2]
        amplitude_range = [0.1, 0.8]

        action_min = np.array([actionSelection_range[0],
                               duration_range[0], 
                               amplitude_range[0]],dtype=np.float32)
        
        action_max = np.array([actionSelection_range[1],
                               duration_range[1], 
                               amplitude_range[1]],dtype=np.float32)

        # Obervation Space min and max
        # target_range = [-0.3, 0.3]
        position_range = [-1.0,2.0]
        velocity_range = [-20,20]

        observation_min = np.array([position_range[0],velocity_range[0]],dtype=np.float32)
        observation_max = np.array([position_range[1],velocity_range[1]],dtype=np.float32)

        action_space = Box(low = action_min, high=action_max,shape=(3,)) 
        observation_space = Box(low = np.array([[-1],[2]],dtype=np.float32),
                                high = np.array([[-20],[20]],dtype=np.float32),shape=(2,1))
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
        extraCost = 0

        return f, extraCost
    
    def get_observation(self,env):
        return np.array([env.x, env.x_dot])
    
    def getZFT(self,action,time): # Later this will become get primatives
        # Because this function changes self.onGoingSubmovements only call it with step==True when using step function
        # in all other cases leave step == false

        # Import and scale all variables are originally 0-1
        actionSelection =  True if action[0]>0.5 else False
        duration = action[1]
        amplitude = action[2]

        extraCost = 0
        
        # add to list
        if actionSelection: # Applied only when step == true
            self.onGoingSubmovements.append([duration, amplitude, time])
            extraCost = -10

        # Remove submovements which are no longer active **ADD LATER TO MAKE MORE EFFICIENT
        # self.onGoingSubmovements = [submov for submov in self.onGoingSubmovements if (submov[0]+submov[3]) >= time]

        x_0b_b, x0_dot_b = self.sumOnGoingSubmovements(time)
        
        # # Submovement Santiy Check (Post-rotation)
        # n = 1000
        # tvec = np.linspace(0,3,n)
        # x0 = np.empty((3,n))
        # x0_dot = np.empty((3,n))
        # for count, t in enumerate(tvec):
        #     self.time = t
        #     x0_tmp,x0_dot_tmp = self.submovement(0.3, 0.3, np.pi*(0.5),0.3,t)
        #     x0[:,count] = np.ndarray.flatten(x0_tmp)
        #     x0_dot[:,count] = np.ndarray.flatten(x0_dot_tmp)

        # fig, ax = plt.subplots()
        # # target = [0.3,0.3,0]
        # ax.plot(x0[0,:],x0[1,:])
        # # ax.plot(target[0],target[1],marker='o',markersize=10,color='k')
        # plt.xlim(0, 0.8)
        # plt.ylim(-1, 1)
        # plt.show()

        # fig, ax = plt.subplots(3)
        # ax[0].plot(x0[0,:])
        # ax[1].plot(x0[1,:])
        # ax[2].plot(x0[2,:])
        # plt.show()

        # fig, ax = plt.subplots(3)
        # ax[0].plot(x0_dot[0,:])
        # ax[1].plot(x0_dot[1,:])
        # ax[2].plot(x0_dot[2,:])
        # plt.show()

        return x_0b_b, x0_dot_b, extraCost
    
    def sumOnGoingSubmovements(self,time):

        x0_tot = np.empty((len(self.onGoingSubmovements),))
        x0_dot_tot = np.empty((len(self.onGoingSubmovements),))

        # Sum up submovements
        for i in range(0,len(self.onGoingSubmovements)):            
            tmp1,tmp2 = self.submovement(self.onGoingSubmovements[i][0],
                                         self.onGoingSubmovements[i][1],
                                         self.onGoingSubmovements[i][2],time)
            x0_tot[i] = tmp1
            x0_dot_tot[i] = tmp2
            
        x_0i_b = np.sum(x0_tot)
        x0_dot_b = np.sum(x0_dot_tot)

        x_0b_b = x_0i_b + self.initial_ib_b

        return x_0b_b, x0_dot_b
    
    def submovement(self,D,A,tStart,time):

        x0,x0_dot,_ = self.getMinJerkTraj_1D(D, A, tStart, time) # Hard code tstart for now

        return x0,x0_dot
         
    def getMinJerkTraj_1D(self,D,A,tstart,time):
        if time <= tstart:
            x = 0.0
            v = 0.0
            a = 0.0
        elif time > tstart+D:
            t = D
            x = A*( (10/D**3)*t**3 + (-15/D**4)*t**4 + (6/D**5)*t**5 )
            v = 0
            a = 0
        else:
            t = time-tstart
            x = A*( (10/D**3)*t**3 + (-15/D**4)*t**4 +  (6/D**5)*t**5 )
            v = A*( (30/D**3)*t**2 + (-60/D**4)*t**3 +  (30/D**5)*t**4 )
            a = A*( (60/D**3)*t   + (-180/D**4)*t**2 + (120/D**5)*t**3 )
        return x,v,a




