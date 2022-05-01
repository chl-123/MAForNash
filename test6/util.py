import random
from decimal import Decimal

from scipy import optimize
import numpy as np
def lin(A):

    #确定c,A,b,Aeq,beq
    c = np.ones(len(A[:,0]))
    # print(c)
    b = np.ones(len(A[:,0]))*-1
    # print(b)

    res = optimize.linprog(c,A,b)
    V=1.0/res.fun

    X=res.x*V
    V = round(V, 5)
    return V,X
def lin2(A,v):

    #确定c,A,b,Aeq,beq
    c = np.ones(len(A[:,0]))
    # print(c)
    b = np.ones(len(A[:,0]))*v
    # print(b)
    Aeq = np.ones((1,len(A[:,0])))
    beq = np.array([1])
    res = optimize.linprog(c,A,b,Aeq,beq)
    return res.x
def stochasticAccept(fitness):
    N = len(fitness)
    maxFit = max(fitness)
    while True:
        ind = int(N * random.random())
        if random.random() <= fitness[ind] / maxFit:
            return ind


import random
from bisect import bisect_left
import numpy as np


"""
Basic roulette wheel selection: O(N)
"""
def basic(fitness):
    '''
    Input: a list of N fitness values (list or tuple)
    Output: selected index
    '''
    sumFits = sum(fitness)
    # generate a random number
    rndPoint = random.uniform(0, sumFits)
    # calculate the index: O(N)
    accumulator = 0.0
    for ind, val in enumerate(fitness):
        accumulator += val
        if accumulator >= rndPoint:
            return ind


"""
Bisecting search roulette wheel selection: O(N + logN)
"""
def bisectSearch(fitness):
    '''
    Input: a list of N fitness values (list or tuple)
    Output: selected index
    '''
    sumFits = sum(fitness)
    # generate a random number
    rndPoint = random.uniform(0, sumFits)
    # calculate the accumulator: O(N)
    accumulator = []
    accsum = 0.0
    for fit in fitness:
        accsum += fit
        accumulator.append(accsum)
    return bisect_left(accumulator, rndPoint)   # O(logN)

# A = np.array([
#     [7.99,7.97,7.90,7.70,7.23],
#     [7.49,7.80,8.00,8.02,7.69],
#     [6.99,7.64,8.11,8.32,8.14],
#     [6.49,7.47,8.22,8.62,8.59],
#     [5.99,7.31,8.32,8.93,9.04]
# ])
# A=np.array(
#     [
#         [1, 0 ,0],
#         [0, 1, 0],
#         [0, 0, 1]
#     ]
# )
A=np.array(
    [
        [3, 1 ,6],
        [0, 0, 4],
        [1, 2, 5]
    ]
)
# B=np.transpose(A)
# A=np.array(
#     [
#         [1, 2 ,0],
#         [0, 1, 2],
#         [2, 0, 1]
#     ]
# )
A=np.array(
    [
        [1, 4 ,5],
        [4, 0, 5],
        [3, 3, 6]
    ]
)
# v1,x=lin(-A)
# # v1 =round(v1, 5)
# print(v1,x)
# B=np.transpose(A)
# print(B)
# y=lin2(B,v1)
# print(y)
# print(B)
# # print(Aeq)
# print(fun)
# print(x)
