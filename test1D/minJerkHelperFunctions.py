import numpy as np    
    
def submovement(D,A,tStart,time):
        # D = sub[0]
        # A = sub[1]
        # tStart = sub[2]
        # time = sub[3]

        if time <= tStart:
            x = 0.0
            v = 0.0
        elif time > tStart+D:
            t = D
            x = A*( (10/D**3)*t**3 + (-15/D**4)*t**4 + (6/D**5)*t**5 )
            v = 0.0
        else:
            t = time-tStart
            x = A*( (10/D**3)*t**3 + (-15/D**4)*t**4 +  (6/D**5)*t**5 )
            v = A*( (30/D**3)*t**2 + (-60/D**4)*t**3 +  (30/D**5)*t**4 )
            
        return x,v