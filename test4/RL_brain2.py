"""
The DQN improvement: Prioritized Experience Replay (based on https://arxiv.org/abs/1511.05952)

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""
import collections
import random

import numpy as np
import tensorflow as tf
import util

np.random.seed(1)
tf.set_random_seed(1)





class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    # beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error
    # capacity=1000
    def __init__(self, capacity,n_features,batch_size):
        # capacity=int(capacity)
        self.batch_size=batch_size
        self.capacity=capacity
        self.priorities = collections.deque(maxlen=10000)
        self.beta = 0.4
        self.memory_counter = 0
        self.memory = np.zeros((10000, n_features * 2 + 3))
    def store(self, transition):

        # replace the old memory with new memory

        index = self.memory_counter % 10000
        self.memory[index, :] = transition
        self.priorities.append(max(self.priorities, default=1))
        self.memory_counter += 1

    def scaled_prob(self):
        # probability updates
        P = np.array(self.priorities, dtype=np.float64)
        P /= P.sum()
        return P

    def prob_imp(self, prob, beta):
        # return importance
        self.beta = beta
        i = (1 / self.capacity * 1 / prob) ** (-self.beta)
        i /= max(i)
        return i



    def sample(self):
        # find the batch and importance using proportional prioritization
        self.beta = np.min([1., 0.001 + self.beta])

        max_mem = min(self.memory_counter, self.capacity)
        probability = self.scaled_prob()
        idex = np.random.choice(max_mem, self.batch_size, replace=False, p=probability)
        imp = self.prob_imp(probability[idex], self.beta)
        batch_memory = self.memory[idex, :]
        imp=np.reshape(imp,(self.batch_size,1))
        return imp, idex,batch_memory

    def prop_priority(self, i, err, c=1.1, alpha_value=0.7):
        # proportional prioritization
        self.priorities[i] = (np.abs(err) + c) ** alpha_value

    def batch_update(self,index,err):
        i = np.arange(self.batch_size)
        for i in range(self.batch_size):
            idx = index[i]
            self.prop_priority(idx, err[i])

class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            output,
            learning_rate=0.005,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=500,
            memory_size=10000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            prioritized=True,
            sess=None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.output=output
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.prioritized = prioritized    # decide to use double q or not

        self.learn_step_counter = 0

        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if self.prioritized:
            self.memory = Memory(capacity=memory_size,n_features=n_features,batch_size=batch_size)


        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess



        self.cost_his = []

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer, trainable):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names,  trainable=trainable)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.output], initializer=w_initializer, collections=c_names,  trainable=trainable)
                b2 = tf.get_variable('b2', [1, self.output], initializer=b_initializer, collections=c_names,  trainable=trainable)
                out = tf.matmul(l1, w2) + b2
            return out

        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.output], name='Q_target')  # for calculating loss
        if self.prioritized:
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer, True)

        with tf.variable_scope('loss'):
            if self.prioritized:
                self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)    # for updating Sumtree
                self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval))
            else:
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer, False)

    def store_transition(self, s, a, o,r, s_):
        if self.prioritized:    # prioritized replay
            transition = np.hstack((s, [a,o, r], s_))
            self.memory.store(transition)    # have high priority for newly arrived transition

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        actions_value=np.array(actions_value).reshape(self.n_actions,self.n_actions)
        V,X=util.lin(actions_value)
        a=util.stochasticAccept(X)
        o = np.random.randint(0, self.n_actions)
        return a,o
    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.prioritized:
            ISWeights, idex,batch_memory = self.memory.sample()

        q_next, q_eval = self.sess.run(
                [self.q_next, self.q_eval],
                feed_dict={self.s_: batch_memory[:, -self.n_features:],
                           self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        V = []
        for i in batch_index:
            a = np.array([q_next[i]]).reshape(self.n_actions, self.n_actions)
            v, x = util.lin(a)
            V.append(v)
        V = np.array(V)
        q_target[batch_index, eval_act_index] = reward + self.gamma * V

        if self.prioritized:
            _, abs_errors, self.cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features],
                                                    self.q_target: q_target,
                                                    self.ISWeights: ISWeights})
            self.memory.batch_update(idex, abs_errors)     # update priority

        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
    def plot_cost(self):
        import matplotlib.pyplot as plt
        np.save("cost1",self.cost_his)
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()