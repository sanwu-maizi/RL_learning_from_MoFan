import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import copy

np.random.seed(1)
torch.manual_seed(1)


class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.el = nn.Linear(n_feature, n_hidden)
        self.q = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.el(x)
        x = F.relu(x)
        x = self.q(x)
        return x


class DeepQNetwork():
    def __init__(self, n_actions, n_features, n_hidden=20, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=200, memory_size=500, batch_size=32, e_greedy_increment=None, ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0

        self.memory = np.zeros((self.memory_size,n_features*2+2))

        self.loss_func = nn.MSELoss()
        self.cost_his = []

        self._build_net()

    def _build_net(self):
        self.q_eval = Net(self.n_features, self.n_hidden, self.n_actions)
        self.q_target = Net(self.n_features, self.n_hidden, self.n_actions)
        self.optimizer = torch.optim.RMSprop(self.q_eval.parameters(),lr=self.lr)

    def store_transition(self,s,a,r,s_):
        if not hasattr(self,'memory_counter'): self.memory_counter = 0 #如果没有建立数据栈，就创立一个
        transition = np.hstack((s,[a,r],s_)) #堆叠为新的水平数组
        index=self.memory_counter%self.memory_size
        self.memory[index,:]=transition
        self.memory_counter +=1

    def choose_action(self,observation):
        observation = torch.Tensor(observation[np.newaxis, :])
        if np.random.uniform()<self.epsilon:
            actions_value = self.q_eval(observation)
            action = np.argmax(actions_value.data.numpy())
        else:
            action=np.random.randint(0,self.n_actions)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter==0:   #按照预定步数，对数据栈进行更新
            self.q_target.load_state_dict(self.q_eval.state_dict())
            print("\ntarget params replaced\n")

        if self.memory_counter > self.memory_size:      #如果数据栈内数据不足，就选用现有数据直接学习，反之从数据栈中随机筛出数据学习
            sample_index = np.random.choice(self.memory_size,size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index,:]

        q_next,q_eval = self.q_target(torch.Tensor(batch_memory[:,-self.n_features:])),self.q_eval(torch.Tensor(batch_memory[:,:self.n_features]))    #两个式子,前者通过q_target前向传播，获取到当前s_转态的参数，后者前者通过q_eval前向传播，获取到s状态的参数
        q_target = torch.Tensor(q_eval.data.numpy().copy())

        batch_index = np.arange(self.batch_size,dtype=np.int32)
        eval_act_index = batch_memory[:,self.n_features].astype(int)
        reward = torch.Tensor(batch_memory[:,self.n_features+1])  #获取到reward列的参数[没有第二个冒号，所以单指这一列]
        q_target[batch_index,eval_act_index] = reward + self.gamma*torch.max(q_next,1)[0]

        loss = self.loss_func(q_eval,q_target)
        self.optimizer.zero_grad()  #清空梯度
        loss.backward() #通过反向传播获得梯度
        self.optimizer.step()   #通过优化器，根据梯度更新优化参数

        self.cost_his.append(loss)
        self.epsilon =self.epsilon+self.epsilon_increment if self.epsilon<self.epsilon_max else self.epsilon_max
        self.learn_step_counter+=1

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)),self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()








