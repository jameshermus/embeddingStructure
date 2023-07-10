import numpy as np
import matplotlib.pyplot as plt

def modelClassical(env,obs):

    x = obs[0,0]
    x_dot = obs[1,0]
    target = obs[2,0]
    x_0_max = obs[3,0]
    # env.robot.onGoingSubmovements 
    # x_0_max, _ = env.robot.sumOnGoingSubmovements(env.time+env.robot.D_high)

    error = target - x_0_max

    # count = 0
    # N = 100
    # errorVec = np.zeros(N)
    # actionVec = np.zeros(N)
    # for error in np.linspace(-0.5,0.5,N):

    if (abs(error) > env.tolerance_x-0.01) & (env.robot.latency > env.robot.thresholdLatency):
        
        if error > 0.2:
            action = 1
        elif error > 0.01:
            action = 2
        elif error < -0.2:
            action = 4
        elif error < -0.01:
            action = 3
        else: 
            action = 0
            
    else:
        action = 0

    #     errorVec[count] = error
    #     actionVec[count] = action

    #     count += 1

    # plt.figure()
    # plt.plot(errorVec,actionVec)
    # plt.xlabel('Error')
    # plt.ylabel('action')
    # plt.show()

    return action