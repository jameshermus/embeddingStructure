from abc import ABC, abstractmethod
from gym.spaces import Box


class controller(ABC):
    def __init__(self,robot):
        self.nJoints = robot.nJoints
        pass
        
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


class zftController(controller):
    def __init__(self,robot):
        super().__init__(robot)

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

# class submovementControl():
#     def __init__():

#     def get_torque():
#         Kq = np.diag([1,1,1,1,1,1,1])
#         Bq = 0.1*Kq
#         q,q_dot = self.get_robotStates(type='np')
#         q0 = np.array([self.q_initial]).transpose() # np.zeros((self.nJoints,1))
#         q0_dot = np.zeros((self.nJoints,1))

#         Kx = np.diag([5000,5000,5000])
#         Bx = 0.1*Kx
#         X = np.array([self.get_ee_position()]).transpose()
#         jac_t_fn = self.get_trans_jacobian()
#         X_dot = np.matmul(jac_t_fn,q_dot)
#         X0,X0_dot = self.getZFT(action)
        
#         tau = np.matmul(jac_t_fn.transpose(), np.matmul(Kx,(X0-X)) + np.matmul(Bx,(X0_dot-X_dot)) ) + np.matmul(Kq,(q0-q)) + np.matmul(Bq,(q0_dot-q_dot))

#         return tau






