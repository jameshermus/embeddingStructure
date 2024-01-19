
import itertools
import time
import numpy as np
import itertools

x = [(1,2),(3,4),(5,6),(7,8)]
y = 1
z = list(zip(x,itertools.repeat(y)))

x+y

def fun1(a,b,c):
    return fun2(a,b,c)

def fun2(a,b,c):
    return a+b+c

start_time = time.time()
N = 1_000_000
test_list = [(1,2,3),(4,5,6),(7,8,10)]
squares = itertools.starmap(fun1, test_list)
x=list(squares)

# x = np.zeros(N)
# for i in range(N):
#     x[i] = fun1(i,2)

total_time_multi = time.time() - start_time
print(f"Took {total_time_multi:.8f}s for learn")

# print(x)

