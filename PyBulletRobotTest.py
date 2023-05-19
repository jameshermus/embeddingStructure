# Import GYM stuff
import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
from scipy.spatial.transform import Rotation as R

# Import helpers
import os,time, sys
import numpy as np
import random

import pybullet as p
import pybullet_data
import math

from robot import robot, robot_iiwa
from controller import controller, zftController

class PyBulletRobotTest(Env):
    def __init__(self,renderType):

        self.time = 0
        self.renderType = renderType

        # Create enviornment
        if self.renderType:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        # Get rootpath
        self.urdfRootPath=pybullet_data.getDataPath()
        self.q_initial = np.array([0,1/4,0,-1/2,0,1/4,-1/4]) * np.pi
        self.q_ititial = self.q_initial.tolist()

        # Add panda, table, box, and object
        self.robot = robot_iiwa(p)
        
        # Setup enviornment
        self.get_enviornment(p)

        # Initialize controller
        self.controller = zftController(self.robot)

        # Set target position
        self.target = np.array([self.robot.x_initial]).transpose()+np.array([[0.2,0.2,0]]).transpose() # Hard code target for first pass

        # Define Spaces
        # Target (x,y,z) extracted from step()
        # Submovement (duration, amplitude, direction) extracted in getZFT()

        self.action_space, self.observation_space = self.controller.define_spaces()

        # q,q_dot = self.get_robotStates(type ='list')
        # self.state = {'jointPostion':q,'jointVelocity':q_dot}

        pass

    def get_enviornment(self,p):    
        self.tableUid = p.loadURDF(os.path.join(self.urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])
        self.trayUid = p.loadURDF(os.path.join(self.urdfRootPath, "tray/traybox.urdf"),basePosition=[0.65,0,0])
        self.objectUid = p.loadURDF(os.path.join(self.urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=[0.7,0,0.1])
        
        p.setGravity(0,0,0) # set gravity

        # Change camera view
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        
        # Time Step
        self.timeStep = 1/500
        p.setTimeStep(self.timeStep)

        self.timeMax = 3

    def step(self,action):

        tau = self.controller(self.state, action)
        p.setJointMotorControlArray(self.robotUid, range(self.nJoints),controlMode=p.TORQUE_CONTROL, forces = tau)
        
        # Step simulation
        p.stepSimulation()
        self.time += self.timeStep

        # Update State
        q,q_dot = self.get_robotStates(type ='list')
        self.state['jointPostion'] = q
        self.state['jointVelocity'] = q_dot

        # Observation
        observation = np.array([self.get_ee_position()]).transpose()
            
        # Assign Reward
        reward = -1*np.linalg.norm(target-observation) # distance cost per time step
            
        # Simulate Real time
        if self.renderType:
            sleep = 1
            cur = time.perf_counter()
            while(sleep):
	            if(time.perf_counter() >= cur + self.timeStep):
		            sleep = 0

        # Initialize info
        info = {}
     
        # Let simulation run a fixed number of time steps
        if self.time >= self.timeMax:
                done = True
        else:
                done = False
        
        return observation, reward, done, info

    
    def render(self):
        pass

    def reset(self):
        for i in range(self.robot.nJoints):
            p.resetJointState(self.robot.robotUid, i, targetValue = self.robot.q_initial[i], targetVelocity = 0)

        q, q_dot = self.robot.get_robotStates(p)

        self.time = 0

        # return self.state
        observation = np.array([self.get_ee_position()]).transpose()

        return observation
    
    def close(self):
         p.disconnect()