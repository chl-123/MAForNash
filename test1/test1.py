import numpy as np
N=8
M=8
target_num=4
Pf=0.5 # 火力单元被敌机发现的概率
Q1=0.6 # 火力单元被敌机毁伤的概率
P1=0.2 # 火力单元捕捉到飞机的概率
P2=0.5 # 火力单元的命中概率
w=[0.2,0.5,0.6,0.5]#保护目标权重
w=np.array(w)
Pf = np.ones((target_num)) * Pf
Q1 = np.ones((target_num)) * Q1
P1 = np.ones((target_num)) * P1
P2 = np.ones((target_num)) * P2
def createAction(N):
    actions = []
    for i in range(0,N):
        for j in range(1,N):
            for k in range(1,N):
                for l in range(1, N):
                    if (i + j + k +l== N):
                        a = [i, j, k,l]
                        actions.append(a)

    return actions
def Dl(Pf,Q1,P1,P2,x):
    D=(1-(1-Pf*Q1)*P1*P2)**x
    return D
def P(D,y,w):
    p=((1-D)*y)/w
    return p
def Step(s,action):
    ActionList = createAction(N)
    x = ActionList[action]
    d = Dl(Pf, Q1, P1, P2, x)
    y = ActionList[7]
    p = P(d, y, w)

    reward=sum(p)-sum(s)
    s_=p
    return s_, reward
a=createAction(N)
print(len(a))