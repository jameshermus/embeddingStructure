import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class modelClassical(ABC):
    def __init__(self,controllerType,tolerance_x, submovementParam = None):
        self.controllerType = controllerType
        self.tolerance_x = tolerance_x
        self.reset()
        pass

    @abstractmethod
    def predict(self,obs):
        # Each implamentation of the define_space method must define the action and observation spaces
        # This will very with diffrent types of x0 vs primative controls
        return action, extra
    
    @abstractmethod
    def reset(self):
        pass
    
class modelClassical_f(modelClassical):
    def __init__(self,controllerType,tolerance_x,submovementParam = None):
        super().__init__(controllerType, tolerance_x,submovementParam = None)

    def predict(self,obs):

        x = obs[0]
        x_dot = obs[1]
        target = obs[2]

        error = target - x

        if (self.nAction_0 > 0 and self.nAction_1>0 and self.nAction_0 == self.nAction_1):
            action = 2
        elif (abs(error) > target/2):
            action = 0
            self.nAction_0 += 1
        elif (abs(error) < target/2):
            action = 1
            self.nAction_1 += 1
        else:
            action = 2

        extra = []

        return action, extra
    
    def reset(self):
        self.nAction_0 = 0
        self.nAction_1 = 0
        pass
    
    
class modelClassical_x0(modelClassical):
    def __init__(self,controllerType,tolerance_x,submovementParam = None):
        super().__init__(controllerType,tolerance_x,submovementParam = None)

    def predict(self,obs):

        x = obs[0]
        x_dot = obs[1]
        target = obs[2]

        action = np.array([target],dtype='float32')

        extra = []

        return action, extra
    
    def reset(self):
        pass
    
class modelClassical_submovement(modelClassical):
    def __init__(self,controllerType,tolerance_x, submovementParam):
        super().__init__(controllerType,tolerance_x, submovementParam)
        self.thresholdLatency = submovementParam['thresholdLatency']
        self.latencyClassic = self.thresholdLatency + 1
        self.D_high = submovementParam['D_high']
        self.A_high = submovementParam['A_high']
        self.A_low = submovementParam['A_low']

    def predict(self,obs):

        x = obs[0]
        x_dot = obs[1]
        target = obs[2]
        x_0_max = obs[3]

        error = target - x_0_max

        if (abs(error) > self.tolerance_x-0.01) & (self.latencyClassic > self.thresholdLatency):
            
            if error > self.A_high:
                action = 1
                self.latencyClassic = 0
            elif error > self.A_low:
                action = 2
                self.latencyClassic = 0
            elif error < -self.A_high:
                action = 3
                self.latencyClassic = 0
            elif error < -self.A_low:
                action = 4
                self.latencyClassic = 0
            else: 
                action = 0
                self.latencyClassic += 1
                
        else:
            action = 0
            self.latencyClassic += 1

        extra = []

        return action, extra
    
    def reset(self):
        pass