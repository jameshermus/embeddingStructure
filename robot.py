from abc import ABC, abstractmethod
import numpy as np
import os
import pybullet_data

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
            
        self.x_initial = self.get_ee_position(p)

        # Disable max force
        maxForces = np.zeros(self.nJoints).tolist()
        p.setJointMotorControlArray(self.robotUid, range(self.nJoints),controlMode=p.VELOCITY_CONTROL, forces=maxForces)

        # Set up torque control
        tau = np.zeros(self.nJoints).tolist()
        p.setJointMotorControlArray(self.robotUid, range(self.nJoints),controlMode=p.TORQUE_CONTROL, forces = tau)
        pass

    def get_ee_position(self,p):
        return p.getLinkState(self.robotUid, self.ee_id)[0]
    
    def get_robotStates(self,p):
        joint_state = p.getJointStates(self.robot.robotUid, range(self.robot.nJoints))
        q = [state[0] for state in joint_state]
        q_dot = [state[1] for state in joint_state]
        q = np.array([q])
        q_dot = np.asarray([q_dot])

        q = q.transpose()
        q_dot = q_dot.transpose()

        return q,q_dot
    
    def get_trans_jacobian(self,p):
        q,_ = self.get_robotStates()
        jac_t_fn, jac_r_fn = p.calculateJacobian(bodyUniqueId = self.robotUid,
                                                 linkIndex = self.ee_id,
                                                 localPosition = self.relative_ee,
                                                 objPositions = q, 
                                                 objVelocities = self.zeros,
                                                 objAccelerations = self.zeros)
        return np.array(jac_t_fn)
    
    @abstractmethod
    def define_spaces(self):
        # Each implamentation of the define_space method must define the action and observation spaces
        # This will very with diffrent types of x0 vs primative controls
        return action_space, obseration_space

    @abstractmethod
    def get_torque(self,action):
        # Each implamentation must define the get_torque method. This method determins the torque to be produced
        # at each time step.

        return tau


class robot_iiwa_zftController(robot_iiwa):
    def __init__(self,p):
        super().__init__(p)

    def define_spaces(self):
        action_space = Box(0,1,shape=(3,)) 
        observation_space = Box(0,1,shape=(3,1))
        return action_space, observation_space

    def get_torque(p, action):
        Kq = np.diag([1,1,1,1,1,1,1])
        Bq = 0.1*Kq
        q,q_dot = p.get_robotStates(type='np')
        q0 = np.array([self.q_initial]).transpose() # np.zeros((self.nJoints,1))
        q0_dot = np.zeros((self.nJoints,1))

        Kx = np.diag([5000,5000,5000])
        Bx = 0.1*Kx
        jac_t_fn = self.robot.get_trans_jacobian(p)
        X, X_dot = state[0]
        X0, X0_dot = action[0]
        
        tau = np.matmul(jac_t_fn.transpose(), np.matmul(Kx,(X0-X)) + np.matmul(Bx,(X0_dot-X_dot)) ) + np.matmul(Kq,(q0-q)) + np.matmul(Bq,(q0_dot-q_dot))

        return tau

class robot_iiwa_submovementControl():
    def __init__(self,p):
        super().__init__(p)

    def get_torque():
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

        return tau




