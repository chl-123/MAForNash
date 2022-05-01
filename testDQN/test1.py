from random import random

import numpy as np
observation=np.random.uniform(0,2,3)
print(observation)
def createAction(N):
    actions = []
    for i in range(0,N+1):
        for j in range(0,N+1):
            for k in range(0,N+1):
                if(i+j+k==N):
                    a=[i,j,k]
                    actions.append(a)

    return actions
a=createAction(5)
print(a)
