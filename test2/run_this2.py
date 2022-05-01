from RL_brain import DeepQNetwork

import numpy as np
N=5
M=5
target_num=3
Pf=0.5 # 火力单元被敌机发现的概率
Q1=0.6 # 火力单元被敌机毁伤的概率
P1=0.2 # 火力单元捕捉到飞机的概率
P2=0.5 # 火力单元的命中概率
w=[0.2,0.6,0.6]#保护目标权重
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
def Step(s,action):
    ActionList = createAction(N)
    x = ActionList[action]
    d = Dl(Pf, Q1, P1, P2, x)
    y = ActionList[5]
    p = P(d, y, w)

    reward=sum(p)-sum(s)
    s_=p
    return s_, reward
def run_maze():
    # initial observation
    observation=np.random.uniform(0,1,3)
    step = 0
    while True:
        # fresh env

        # RL choose action based on observation
        action = RL.choose_action(observation)

        # RL take action and get next observation and reward
        observation_, reward= Step(observation,action)
        # print(observation_)
        RL.store_transition(observation, action, reward, observation_)
        print(reward)
        if (step > 200) and (step % 5 == 0):
            # print(action)
            RL.learn()

        # swap observation
        observation = observation_

        # break while loop when end of this episode
        if step > 10000:
            # print(action)
            break
        step += 1

    # end of game


print('game over')



if __name__ == "__main__":
    RL = DeepQNetwork(10, target_num,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.95,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    run_maze()
    RL.plot_cost()