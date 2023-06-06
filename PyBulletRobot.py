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

from robot import robot, robot_iiwa, robot_iiwa_tauController, robot_iiwa_zftController, robot_iiwa_submovementControl

class PyBulletRobot(Env):
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
        # self.robot = robot_iiwa_tauController(p)
        # self.robot = robot_iiwa_zftController(p)
        self.robot = robot_iiwa_submovementControl(p)

        # Setup enviornment
        self.get_enviornment(p)

        # Define Spaces
        # Target (x,y,z) extracted from step()
        # Submovement (duration, amplitude, direction) extracted in getZFT()

        self.action_space, self.observation_space = self.robot.define_spaces()

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
        self.timeStep = 1/750
        p.setTimeStep(self.timeStep)

        self.timeMax = 0.5

    def step(self,action):

        tau,extraCost = self.robot.get_torque(p,action,self.time)
        p.setJointMotorControlArray(self.robot.robotUid, range(self.robot.nJoints),controlMode=p.TORQUE_CONTROL, forces = tau)
        
        # Step simulation
        p.stepSimulation()
        self.time += self.timeStep

        # Update State
        # q,q_dot = self.get_robotStates(type ='list')
        # self.state['jointPostion'] = q
        # self.state['jointVelocity'] = q_dot

        # Observation
        observation = self.get_observation(p)
            
        # Assign Reward
        reward = -1*np.linalg.norm(self.target_tb_b-self.robot.get_ee_position(p)) + extraCost # distance cost per time step
            
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
        # elif (np.abs(reward) < 0.2):
        #         done = True
        else:
                done = False
        
        return observation, reward, done, info

    
    def render(self,env):
        pass

    def reset(self):
        for i in range(self.robot.nJoints):
            p.resetJointState(self.robot.robotUid, i, targetValue = self.robot.q_initial[i], targetVelocity = 0)

        self.time = 0 # Reset time

        # Convert target abstract to vector 
        # self.target_ti_b = self.observation_space.sample()[0:3] # For now use target (t) which is plus or minus 0.25 m from initial end-effector position (i) represented in the base frame (b)
        self.target_ti_b = np.array([[0.2],[0.0],[0.0]])

        # Define self.target_tb_b # vector from the origin of the base frame to the target expressed in base coordinates
        self.target_tb_b = self.target_ti_b + self.robot.initial_ib_b  # Hard code target for first pass

        observation = self.get_observation(p)

        return observation
    
    def close(self):
         p.disconnect()

    def get_observation(self,p):
        # obervation uses the target reletive to the inital end-effector pose
        observation = np.concatenate((self.target_ti_b,self.robot.get_ee_position(p)),axis=0)
        return observation
