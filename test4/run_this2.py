from statistics import mean

from RL_brain import DeepQNetwork

import numpy as np
N=5
M=5
target_num=3
Pf=0.5 # 火力单元被敌机发现的概率
Q1=0.6 # 火力单元被敌机毁伤的概率
P1=0.2 # 火力单元捕捉到飞机的概率
P2=0.5 # 火力单元的命中概率
w=[0.2,0.5,0.6]#保护目标权重
w=np.array(w)
Pf = np.ones((target_num)) * Pf
Q1 = np.ones((target_num)) * Q1
P1 = np.ones((target_num)) * P1
P2 = np.ones((target_num)) * P2


def plot_cost(reward):
    import matplotlib.pyplot as plt
    np.save("reward", reward)
    plt.plot(np.arange(len(reward)), reward)
    plt.ylabel('reward')
    plt.xlabel('training steps')
    plt.show()
def createAction(N):
    actions = []
    for i in range(0,N+1):
        for j in range(0,N+1):
            for k in range(0,N+1):
                if(i+j+k==N):
                    a=[i,j,k]
                    actions.append(a)

    return actions
def Dl(Pf,Q1,P1,P2,x):
    D=(1-(1-Pf*Q1)*P1*P2)**x
    return D
def P(D,y,w):
    p=((1-D)*y)/w
    return p
def Step(s,a,o):
    ActionList = createAction(N)
    x = ActionList[a]
    d = Dl(Pf, Q1, P1, P2, x)
    y = ActionList[o]
    p = P(d, y, w)
    #
    reward=sum(p)-sum(s)
    s_=p
    return s_, reward
def run_maze():
    # initial observation
    observation=np.random.uniform(0,2,3)
    step = 0
    r=[]
    r2=[]
    while True:
        # fresh env

        # RL choose action based on observation
        a,o = RL.choose_action(observation)
        # RL take action and get next observation and reward
        observation_, reward= Step(observation,a,o)
        # print(observation_)
        RL.store_transition(observation, a, o,reward, observation_)
        r.append(reward)
        r2.append(mean(r))

        if (step > 1000) and (step % 5 == 0):
            # print(action)
            RL.learn()


        # swap observation
        observation = observation_
        print(step)
        # break while loop when end of this episode
        if step > 20000:
            # print(action)
            break
        step += 1

    return r2
    # end of game





if __name__ == "__main__":
    RL = DeepQNetwork(21, target_num,
                      output=21*21,
                      learning_rate=0.05,
                      reward_decay=0.8,
                      e_greedy=0.92,
                      replace_target_iter=200,
                      memory_size=10000,

                      )
    r=run_maze()
    # plot_cost(r)
    RL.plot_cost()