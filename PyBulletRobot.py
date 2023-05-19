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
        urdfRootPath=pybullet_data.getDataPath()
        self.q_initial = np.array([0,1/4,0,-1/2,0,1/4,-1/4]) * np.pi
        self.q_ititial = self.q_initial.tolist()

        # Add panda, table, box, and object
        # self.robotUid = p.loadURDF(os.path.join(pybullet_data.getDataPath(),"franka_panda/panda.urdf"),useFixedBase =True)
        self.robotUid = p.loadURDF(os.path.join(pybullet_data.getDataPath(),"kuka_iiwa/model.urdf"), useFixedBase=True)
        
        tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])
        trayUid = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),basePosition=[0.65,0,0])
        objectUid = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=[0.7,0,0.1])
        
        p.setGravity(0,0,0) # set gravity

        # Change camera view
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])

        # Set robot initial postion
        # p.resetJointState(bodyUniqueId=pandaUid,jointIndex=, tragetValue= targetVelocity=)

        # Set up torque control
        self.nJoints = p.getNumJoints(self.robotUid)
        self.ee_id = 6
        self.zeros = [0.0] * self.nJoints
        self.relative_ee = [0, 0, 0]

        # Set initial robot position
        for i in range(0,self.nJoints):
            # p.setJointMotorControl2(self.robotUid, i, p.POSITION_CONTROL,self.q_initial[i])
            p.resetJointState(self.robotUid, i, targetValue = self.q_initial[i], targetVelocity = 0)
        
        self.x_initial = self.get_ee_position()

        # Disable max force
        maxForces = np.zeros(self.nJoints).tolist()
        p.setJointMotorControlArray(self.robotUid, range(self.nJoints),controlMode=p.VELOCITY_CONTROL, forces=maxForces)

        # Set up torque control
        tau = np.zeros(self.nJoints).tolist()
        p.setJointMotorControlArray(self.robotUid, range(self.nJoints),controlMode=p.TORQUE_CONTROL, forces = tau)

        # Time Step
        self.timeStep = 1/500
        p.setTimeStep(self.timeStep)

        self.timeMax = 3

        # Define Spaces
        # Target (x,y,z) extracted from step()
        # Submovement (duration, amplitude, direction) extracted in getZFT()
        self.action_space = Box(0,1,shape=(3,)) 

        self.observation_space = Box(0,1,shape=(3,1))

        q,q_dot = self.get_robotStates(type ='list')
        self.state = {'jointPostion':q,'jointVelocity':q_dot}

        pass

    def step(self,action):

        target = np.array([self.x_initial]).transpose()+np.array([[0.2,0.2,0]]).transpose() # Hard code target for first pass
        reward = 0

        while self.time <= self.timeMax:
            # Define control torque
            tau = self.controller(action)
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
            reward += -1*np.linalg.norm(target-observation) # distance cost per time step
            
            # Simulate Real time
            if self.renderType:
                sleep = 1
                cur = time.perf_counter()
                while(sleep):
	                if(time.perf_counter() >= cur + self.timeStep):
		                sleep = 0



        # Initialize info
        info = {}
     
        # # Let simulation run a fixed number of time steps
        # if self.time >= self.timeMax:
        #         done = True
        # else:
        #         done = False

        # First pass run one step
        done = True
        
        return observation, reward, done, info

    
    def render(self):
        pass

    def reset(self):
        for i in range(self.nJoints):
            p.resetJointState(self.robotUid, i, targetValue = self.q_initial[i], targetVelocity = 0)

        q,q_dot = self.get_robotStates(type ='list')
        self.state['jointPostion'] = q
        self.state['jointVelocity'] = q_dot

        self.time = 0

        # return self.state
        observation = np.array([self.get_ee_position()]).transpose()
        return observation
    
    def close(self):
         p.disconnect()

    def controller(self,action):
        Kq = np.diag([1,1,1,1,1,1,1])
        Bq = 0.1*Kq
        q,q_dot = self.get_robotStates(type='np')
        q0 = np.array([self.q_initial]).transpose() # np.zeros((self.nJoints,1))
        q0_dot = np.zeros((self.nJoints,1))

        Kx = np.diag([5000,5000,5000])
        Bx = 0.1*Kx
        X = np.array([self.get_ee_position()]).transpose()
        jac_t_fn = self.get_trans_jacobian()
        X_dot = np.matmul(jac_t_fn,q_dot)
        X0,X0_dot = self.getZFT(action)

        tau = np.matmul(jac_t_fn.transpose(), np.matmul(Kx,(X0-X)) + np.matmul(Bx,(X0_dot-X_dot)) ) + np.matmul(Kq,(q0-q)) + np.matmul(Bq,(q0_dot-q_dot))

        tau = tau.flatten().tolist()
        return tau
    
    def get_ee_position(self):
        return p.getLinkState(self.robotUid, self.ee_id)[0]

    def get_trans_jacobian(self):
        q,_ = self.get_robotStates(type = 'list')
        jac_t_fn, jac_r_fn = p.calculateJacobian(bodyUniqueId = self.robotUid,
                                                 linkIndex = self.ee_id,
                                                 localPosition = self.relative_ee,
                                                 objPositions = q, 
                                                 objVelocities = self.zeros,
                                                 objAccelerations = self.zeros)
        return np.array(jac_t_fn)
    
    def get_robotStates(self,type = 'np'):
        joint_state = p.getJointStates(self.robotUid, range(self.nJoints))
        q = [state[0] for state in joint_state]
        q_dot = [state[1] for state in joint_state]
        if type == 'np':
            q = np.array([q])
            q_dot = np.asarray([q_dot])

            q = q.transpose()
            q_dot = q_dot.transpose()
        return q,q_dot

    def getZFT(self,action): # Later this will become get primatives
        # Import and scale all variables are originally 0-1
        duration = action[0]*2+0.2
        amplitude = action[1]*1+0.01
        direction = action[2]*2*np.pi

        x0,x0_dot = self.submovement(duration, amplitude, direction)

        # # # Submovement Santiy Check (Post-rotation)
        # n = 1000
        # tvec = np.linspace(0,3,n)
        # x0 = np.empty((3,n))
        # for count, t in enumerate(tvec):
        #     self.time = t
        #     x0_tmp,x0_dot_tmp = self.submovement(duration, amplitude, direction)
        #     x0[:,count] = np.ndarray.flatten(x0_tmp)

        # fig, ax = plt.subplots()
        # target = [0.3,0.3,0]
        # ax.plot(x0[0,:],x0[1,:])
        # ax.plot(target[0],target[1],marker='o',markersize=10,color='k')
        # plt.xlim(0, 0.8)
        # plt.ylim(-1, 1)
        # plt.show()

        return x0, x0_dot 
    
    def submovement(self,D,A,Direction):
        x0_1d,x0_dot_1d,_ = self.getMinJerkTraj_1D(D,A,0) # Hard code tstart for now
        rotMatrix = np.reshape(np.array([[np.cos(Direction), -np.sin(Direction)], 
                              [np.sin(Direction),  np.cos(Direction)]]),(2,2))
        X0 = np.matmul(rotMatrix,np.array([[x0_1d],[0.]])) 
        X0_dot = np.matmul(rotMatrix,np.array([[x0_dot_1d],[0]]))

        X0 = np.insert(X0,2,0,axis=0) + np.array([self.x_initial]).transpose() # add initial condition for now
        X0_dot = np.insert(X0_dot,2,0,axis=0)

        # Submovement Sanity Check (pre-rotation)
        # x0_1d = []
        # x0_dot_1d = []
        # tvec = np.linspace(0,3,1000)
        # for count, t in enumerate(tvec):
        #     self.time = t
        #     x,xd,_ = self.getMinJerkTraj_1D(D,A,0) # Hard code tstart for now
        #     x0_1d.append(x)
        #     x0_dot_1d.append(xd)

        # fig, ax = plt.subplots()
        # ax.plot(tvec,x0_1d)
        # plt.show()

        # fig, ax = plt.subplots()
        # ax.plot(tvec,x0_dot_1d)
        # plt.show()

        return X0,X0_dot
         
    def getMinJerkTraj_1D(self,D,A,tstart):
        if self.time <= tstart:
            x = 0.0
            v = 0.0
            a = 0.0
        elif self.time > tstart+D:
            t = D
            x = A*( (10/D**3)*t**3 + (-15/D**4)*t**4 + (6/D**5)*t**5 )
            v = 0
            a = 0
        else:
            t = self.time-tstart
            x = A*( (10/D**3)*t**3 + (-15/D**4)*t**4 +  (6/D**5)*t**5 )
            v = A*( (30/D**3)*t**2 + (-60/D**4)*t**3 +  (30/D**5)*t**4 )
            a = A*( (60/D**3)*t   + (-180/D**4)*t**2 + (120/D**5)*t**3 )
        return x,v,a