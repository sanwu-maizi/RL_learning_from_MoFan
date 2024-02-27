import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(1)
torch.manual_seed(1)


class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.layer = nn.Linear(n_feature, n_hidden)
        self.all_act = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.layer(x)
        x = torch.tanh(x)
        x = self.all_act(x)


class PolicyGradient:
    def __init__(self, n_actions, n_features, n_hidden=10, learning_rate=0.01, reward_decay=0.95):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self.build_net()

    def build_net(self):
        self.net = Net(self.n_features, self.n_hidden, self.n_actions)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def choose_action(self, observation):
        observation=np.array(observation,dtype=int)
        prob_weights = self.net(observation)  # 正向传播状态获得对应权重
        prob = F.softmax(prob_weights)  # 将动作概率转换为动作分布，总和为1
        action = np.random.choice(range(prob_weights.shape[1]), p=prob.data.numpy().ravel())
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        discounter_ep_re_norm = self._discount_and_norm_rewards()
        obs = torch.Tensor(np.vstack(self.ep_obs))
        acts = torch.Tensor(np.array(self.ep_as))
        vt = torch.Tensor(discounter_ep_re_norm)

        all_act = self.net(obs)  # 通过神经网络获取所有动作的得分。

        neg_log_prob = F.cross_entropy(all_act, acts.long(),
                                       reduce=False)  # 计算负对数似然，即交叉熵。acts.long() 将动作张量转换为长整型，reduce=False 表示不对损失进行求和。
        loss = torch.mean(neg_log_prob * vt)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.ep_obs, self.ep_as, self_rs = [], [], []
        return discounter_ep_re_norm

    def _discount_and_norm_rewards(self):
        discount_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discount_ep_rs[t] = running_add

        discount_ep_rs -= np.mean(discount_ep_rs)  #均值为0
        discount_ep_rs /= np.std(discount_ep_rs)   #归一化

        return discount_ep_rs
