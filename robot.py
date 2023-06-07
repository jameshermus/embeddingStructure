from abc import ABC, abstractmethod
import numpy as np
import os
import pybullet_data

from gym.spaces import Box
import matplotlib.pyplot as plt


# Use to define notation later
# https://drake.mit.edu/doxygen_cxx/group__multibody__quantities.html

class robot(ABC):
    def __init__(self,p):
        self.robotUid = p.loadURDF(os.path.join(pybullet_data.getDataPath(),"kuka_iiwa/model.urdf"), useFixedBase=True)
        self.nJoints = p.getNumJoints(self.robotUid)
        self.relative_ee = [0, 0, 0]
        self.zeros = [0.0] * self.nJoints
        pass

class robot_iiwa(robot):
    def __init__(self,p):
        super().__init__(p)
        self.q_initial = np.array([0,1/4,0,-1/2,0,1/4,-1/4]) * np.pi
        self.q_initial = self.q_initial.tolist()
        
        # Set up torque control
        self.ee_id = 6

        # Set initial robot position
        for i in range(0,self.nJoints):
            # p.setJointMotorControl2(self.robotUid, i, p.POSITION_CONTROL,self.q_initial[i])
            p.resetJointState(self.robotUid, i, targetValue = self.q_initial[i], targetVelocity = 0)
            
        self.initial_ib_b = self.get_ee_position(p)

        # Disable max force
        maxForces = np.zeros(self.nJoints).tolist()
        p.setJointMotorControlArray(self.robotUid, range(self.nJoints),controlMode=p.VELOCITY_CONTROL, forces=maxForces)

        # Set up torque control
        tau = np.zeros(self.nJoints).tolist()
        p.setJointMotorControlArray(self.robotUid, range(self.nJoints),controlMode=p.TORQUE_CONTROL, forces = tau)
        pass

    def get_ee_position(self,p):
        return np.array([p.getLinkState(self.robotUid, self.ee_id)[0]]).transpose()
        
    def get_ee_velocity(self,p):
        return np.array([p.getLinkState(self.robotUid, self.ee_id, computeLinkVelocity=True)[6]]).transpose()
    
    def get_robotStates(self,p):
        joint_state = p.getJointStates(self.robotUid, range(self.nJoints))
        q = [state[0] for state in joint_state]
        q_dot = [state[1] for state in joint_state]
        q = np.array([q])
        q_dot = np.asarray([q_dot])

        q = q.transpose()
        q_dot = q_dot.transpose()

        return q,q_dot
    
    def get_trans_jacobian(self,p):
        q,_ = self.get_robotStates(p)
        jac_t_fn, jac_r_fn = p.calculateJacobian(bodyUniqueId = self.robotUid,
                                                 linkIndex = self.ee_id,
                                                 localPosition = self.relative_ee,
                                                 objPositions = q.flatten().tolist(), 
                                                 objVelocities = self.zeros,
                                                 objAccelerations = self.zeros)
        return np.array(jac_t_fn)
    
    @abstractmethod
    def define_spaces(self):
        # Each implamentation of the define_space method must define the action and observation spaces
        # This will very with diffrent types of x0 vs primative controls
        return action_space, obseration_space

    @abstractmethod
    def get_torque(p,action,time):
        # Each implamentation must define the get_torque method. This method determins the torque to be produced
        # at each time step.

        return tau, extraCost

class robot_iiwa_tauController(robot_iiwa):
    def __init__(self,p):
        super().__init__(p)

    def define_spaces(self):
        action_space = Box(-320,320,shape=(self.nJoints,)) 
        observation_space = Box(0,1,shape=(3,1))
        return action_space, observation_space

    def get_torque(self,p, action,time):
        tau = action.reshape(self.nJoints,1)
        extraCost = 0
        return tau, extraCost

class robot_iiwa_zftController(robot_iiwa):
    def __init__(self,p):
        super().__init__(p)
        self.priorAction = []

    def define_spaces(self):

        # Obervation Space min and max
        target_range = [-0.3, 0.3]
        position_range = [-1.0,1.0]
        velocity_range = [-100.0,100.0]

        observation_min = np.array([[target_range[0]],[target_range[0]],[target_range[0]],
                                    [position_range[0]],[position_range[0]],[position_range[0]],
                                    [velocity_range[0]],[velocity_range[0]],[velocity_range[0]]],dtype=np.float32)
        observation_max = np.array([[target_range[1]],[target_range[1]],[target_range[1]],
                                    [position_range[1]],[position_range[1]],[position_range[1]],
                                    [velocity_range[1]],[velocity_range[1]],[velocity_range[1]]],dtype=np.float32)
        
        action_space = Box(low = -1, high = 1, shape=(3,1)) 
        observation_space = Box(low = observation_min, high = observation_max, shape = (9,1)) # First 3 target, second 3 current position, third 3 observed speed

        return action_space, observation_space

    def get_torque(self, p, action, time,step = True):
        Kq = np.diag([1,1,1,1,1,1,1])
        Bq = 0.1*Kq
        q,q_dot = self.get_robotStates(p)
        q0 = np.array([self.q_initial]).transpose() # np.zeros((self.nJoints,1))
        q0_dot = np.zeros((self.nJoints,1))

        Kx = np.diag([5000,5000,5000])
        Bx = 0.1*Kx
        jac_t_fn = self.get_trans_jacobian(p)
        X = self.get_ee_position(p)
        X_dot = np.matmul(jac_t_fn,q)

        if(step):
            X0 = action.reshape(3,1)
            self.priorAction = action
        else:
            X0 = self.priorAction.reshape(3,1)

        X0_dot = np.zeros((3,1))
        
        tau = np.matmul(jac_t_fn.transpose(), np.matmul(Kx,(X0-X)) + np.matmul(Bx,(X0_dot-X_dot)) ) + np.matmul(Kq,(q0-q)) + np.matmul(Bq,(q0_dot-q_dot))
        extraCost = 0

        return tau, extraCost

class robot_iiwa_submovementControl(robot_iiwa):
    def __init__(self,p):
        super().__init__(p)
        self.onGoingSubmovements = []

    def define_spaces(self):
        # Action Space Ranges min and max
        actionSelection_range = [0.0, 1.0]
        duration_range = [0.2, 1.0]
        amplitude_range = [0.1, 0.3]
        direction_range = [0, 2*np.pi]

        action_min = np.array([[actionSelection_range[0]],
                               [duration_range[0]], 
                               [amplitude_range[0]]],dtype=np.float32)
                            #    [direction_range[0]]],dtype=np.float32)
        
        action_max = np.array([[actionSelection_range[1]],
                               [duration_range[1]], 
                               [amplitude_range[1]]],dtype=np.float32)
                            #    [direction_range[1]]],dtype=np.float32)

        # Obervation Space min and max
        target_range = [-0.3, 0.3]
        position_range = [-1.0,1.0]
        velocity_range = [-100.0,100.0]

        observation_min = np.array([[target_range[0]],[target_range[0]],[target_range[0]],
                                    [position_range[0]],[position_range[0]],[position_range[0]],
                                    [velocity_range[0]],[velocity_range[0]],[velocity_range[0]]],dtype=np.float32)
        observation_max = np.array([[target_range[1]],[target_range[1]],[target_range[1]],
                                    [position_range[1]],[position_range[1]],[position_range[1]],
                                    [velocity_range[1]],[velocity_range[1]],[velocity_range[1]]],dtype=np.float32)

        action_space = Box(low = action_min, high = action_max,shape = (3,1)) 
        observation_space = Box(low = observation_min, high = observation_max, shape = (9,1)) # First 3 target, second 3 current position, third 3 observed speed
        return action_space, observation_space

    def get_torque(self,p,action,time,step=True):
        Kq = np.diag([1,1,1,1,1,1,1])
        Bq = 0.1*Kq
        q,q_dot = self.get_robotStates(p)
        q0 = np.array([self.q_initial]).transpose() # np.zeros((self.nJoints,1))
        q0_dot = np.zeros((self.nJoints,1))

        Kx = np.diag([5000,5000,5000])
        Bx = 0.1*Kx
        X = self.get_ee_position(p)
        jac_t_fn = self.get_trans_jacobian(p)
        X_dot = np.matmul(jac_t_fn,q_dot)
        X0,X0_dot,extraCost = self.getZFT(action, time, step)
        
        tau = np.matmul(jac_t_fn.transpose(), np.matmul(Kx,(X0-X)) + np.matmul(Bx,(X0_dot-X_dot)) ) + np.matmul(Kq,(q0-q)) + np.matmul(Bq,(q0_dot-q_dot))

        return tau, extraCost
    
    def getZFT(self,action,time,step=True): # Later this will become get primatives
        # To allow for getZFT to be called by the simulation when no new actions are passed step
            
        extraCost = 0

        if(step):
            # Import and scale all variables are originally 0-1
            actionSelection =  True if action[0]>0.5 else False
            duration = action[1]
            amplitude = action[2]
            direction = 0 #action[3]
        
            # add to list
            if actionSelection: # Applied only when step == true
                self.onGoingSubmovements.append([duration, amplitude, direction, time])
                extraCost = -1
        

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

        x0_tot = np.empty((len(self.onGoingSubmovements),3))
        x0_dot_tot = np.empty((len(self.onGoingSubmovements),3))

        # Sum up submovements
        for i in range(0,len(self.onGoingSubmovements)):            
            tmp1,tmp2 = self.submovement(self.onGoingSubmovements[i][0],
                                         self.onGoingSubmovements[i][1],
                                         self.onGoingSubmovements[i][2],
                                         self.onGoingSubmovements[i][3],time)
            x0_tot[i,:] = tmp1.flatten()
            x0_dot_tot[i,:] = tmp2.flatten()
            
        x_0i_b = np.sum(x0_tot,0).reshape((3,1)) 
        x0_dot_b = np.sum(x0_dot_tot,0).reshape((3,1))

        x_0b_b = x_0i_b + self.initial_ib_b

        return x_0b_b, x0_dot_b
    
    def submovement(self,D,A,Direction,tStart,time):
        x0_1d = np.zeros((2,1))
        x0_dot_1d = np.zeros((2,1))
        x0_1d[0],x0_dot_1d[0],_ = self.getMinJerkTraj_1D(D, A, tStart, time) # Hard code tstart for now
        rotMatrix = np.reshape(np.array([[np.cos(Direction), -np.sin(Direction)], 
                              [np.sin(Direction),  np.cos(Direction)]]), (2,2))
        
        X0 = np.matmul(rotMatrix, x0_1d) 
        X0_dot = np.matmul(rotMatrix, x0_dot_1d)

        X0 = np.insert(X0, 2, 0, axis=0)
        X0_dot = np.insert(X0_dot, 2, 0, axis=0)

        # Submovement Sanity Check (pre-rotation)
        # tvec = np.linspace(0,3,1000)
        # x0_1d = np.zeros((1000,1))
        # x0_dot_1d = np.zeros((1000,1))
        # for count, t in enumerate(tvec):
        #     self.time = t
        #     x,xd,_ = self.getMinJerkTraj_1D(D,A,0.3,t) # Hard code tstart for now
        #     x0_1d[count,0] = x
        #     x0_dot_1d[count,0] = xd

        # fig, ax = plt.subplots()
        # ax.plot(tvec,x0_1d)
        # plt.show()

        # fig, ax = plt.subplots()
        # ax.plot(tvec,x0_dot_1d)
        # plt.show()

        return X0,X0_dot
         
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




