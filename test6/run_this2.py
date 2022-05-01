import random
from statistics import mean

from RL_brain import DeepQNetwork
import numpy as np
N=5
M=5
target_num=5
num=3
P1=0.3
P2=0.8
w=[0.2,0.5,0.6]#保护目标权重
w=np.array(w)

P1 = np.ones((num)) * P1
P2 = np.ones((num)) * P2
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
def Step(observation,a,o):

    x = ActionList[a]
    q= Q(P1,x)
    y = ActionList[o]
    s =S(q,y,P2)
    v1,v2=V(s, w)
    reward=v1-v2-observation[3]-observation[4]
    k=np.zeros(5)
    k[0:3]=s
    k[3:4]=v1
    k[4:5]=v2
    return k,reward


def plot_cost(reward):
    import matplotlib.pyplot as plt
    np.save("reward", reward)
    plt.plot(np.arange(len(reward)), reward)
    plt.ylabel('reward')
    plt.xlabel('training steps')
    plt.show()

def run_maze():
    # initial observation
    observation=np.random.uniform(0,1,5)
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

        if (step > 10000) and (step % 10 == 0):
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
                      e_greedy=0.90,
                      replace_target_iter=200,
                      memory_size=10000,
                      )
    r=run_maze()
    # plot_cost(r)
    RL.plot_cost()