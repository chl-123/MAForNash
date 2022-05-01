import random

import numpy as np
N=5
M=5
target_num=3

P1=0.3
P2=0.8
w=[0.2,0.5,0.6]#保护目标权重
w=np.array(w)

P1 = np.ones((target_num)) * P1
P2 = np.ones((target_num)) * P2
def createAction(N):
    actions = []
    for i in range(0,N+1):
        for j in range(0,N+1):
            for k in range(0,N+1):
                if(i+j+k==N):
                    a=[i,j,k]
                    actions.append(a)

    return actions
def Q(P1,x):
    q=1-(1-P1)**x
    return q
def S(q,y,P2):
    p=(1-(1-P2)**y)*(1-q)
    return p
def V(s,w):
    v1=sum((1-s)*w)
    v2 = sum(s* w)
    return v1,v2

ActionList=createAction(N)
def Step(a,o,v):

    x = ActionList[a]
    q= Q(P1,x)
    y = ActionList[o]
    s =S(q,y,P2)
    v1,v2=V(s, w)
    reward=v1-v2-v
    k=np.zeros(5)
    k[0:3]=s
    k[3:4]=v1
    k[4:5]=v2
    return k,reward
c,reward=Step(2,12,0.2)
a=random.random()
print(a)

