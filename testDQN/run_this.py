import numpy as np

from maze_env import Maze
from RL_brain import DeepQNetwork

def plot_cost(reward):
    import matplotlib.pyplot as plt
    np.save("reward", reward)
    plt.plot(np.arange(len(reward)), reward)
    plt.ylabel('reward')
    plt.xlabel('training steps')
    plt.show()
def run_maze():
    step = 0
    r = []
    for episode in range(10):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            # print(observation_)
            RL.store_transition(observation, action, reward, observation_)
            r.append(reward)
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()
    return r

if __name__ == "__main__":
    # maze game
    env = Maze()
    # print(env.n_actions)
    # print(env.n_features)
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    r=run_maze()
    plot_cost(r)
    # RL.plot_cost()