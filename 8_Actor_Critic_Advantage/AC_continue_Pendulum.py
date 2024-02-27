import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(1)
torch.manual_seed(1)


class Actor_Net(nn.Module):
    def __init__(self, n_features, n_hidden, n_outputs):
        super(Actor_Net, self).__init__()
        self.l1 = nn.Linear(n_features, n_hidden)
        self.mu = nn.Linear(n_hidden, n_outputs)
        self.sigma = nn.Linear(n_hidden, n_outputs)

    def forward(self,x):
        x = self.l1(x)
        x = F.relu(x)
        mu = self.mu(x)
        mu = torch.tanh(mu)
        sigma = self.sigma(x)
        sigma = F.softplus(sigma)

        return mu, sigma


class Actor(object):
    def __init__(self, n_features, action_bound, n_hidden=30, lr=0.0001):
        self.n_features = n_features
        self.action_bound = action_bound
        self.n_hidden = n_hidden
        self.lr = lr

        self._build_net()

    def _build_net(self):
        self.actor_net = Actor_Net(self.n_features, self.n_hidden, 1)
        self.actor_net = nn.parallel.DistributedDataParallel(
            self.actor_net,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,  # 设置 broadcast_buffers 为 False
            find_unused_parameters=True
        )
        self.optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.lr)

    def normal_dist(self, s):
        s=np.array(s[0],dtype=float)
        s=torch.Tensor(s[np.newaxis, :])
        mu, sigma = self.actor_net(s)
        mu, sigma = (mu * 2).squeeze(), (sigma + 0.1).squeeze()
        normal_dist = torch.distributions.Normal(mu, sigma)  # 正态分布对象，其中 mu 是均值，sigma 是标准差
        return normal_dist

    def choose_action(self, s):
        normal_list = self.normal_dist(s)
        self.action = torch.clamp(normal_list.sample(), self.action_bound[0], self.action_bound[1])
        #print(f"After conversion: type={type(self.action)}, value={self.action}")
        self.action = self.action.unsqueeze(0)
        #self.action = torch.clamp(normal_list.sample(), self.action_bound[0], self.action_bound[1])
        return self.action

    def learn(self, s, a, td):
        normal_dist = self.normal_dist(s)
        log_prob = normal_dist.log_prob(a)
        exp_v = log_prob * td.float()
        exp_v += 0.01 * normal_dist.entropy()
        loss = -exp_v

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return exp_v


class Critic_Net(nn.Module):
    def __init__(self, n_features, n_hidden, n_outputs):
        super(Critic_Net, self).__init__()
        self.l1 = nn.Linear(n_features, n_hidden)
        self.v = nn.Linear(n_hidden, n_outputs)

    def forward(self,x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.v(x)
        return x


class Critic(object):
    def __init__(self, n_features, n_hidden=30, n_output=1, lr=0.01):
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.lr = lr

        self._build_net()

    def _build_net(self):
        self.critic_net = Critic_Net(self.n_features, self.n_hidden, self.n_output)
        self.critic_net = nn.parallel.DistributedDataParallel(
            self.critic_net,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,  # 设置 broadcast_buffers 为 False
            find_unused_parameters=True
        )
        self.optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.lr)

    def learn(self, s, r, s_):
        s= np.array(s[0], dtype=float)
        #print(f"s conversion: type={type(s)}, value={s}")
        #print(f"s_ conversion: type={type(s_)}, value={s_}")
        s, s_ = torch.Tensor(s[np.newaxis, :]), torch.Tensor(s_[np.newaxis, :])
        v, v_ = self.critic_net(s), self.critic_net(s_)
        v,v_=v.clone(),v_.clone()
        td_error = torch.mean(r + GAMMA * v_ - v)
        loss = td_error ** 2
        torch.autograd.set_detect_anomaly(True)

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        return td_error


MAX_EPISODE = 1000
MAX_EP_STEPS = 200
DISPLAY_REWARD_THRESHOLD = -100
RENDER = False
GAMMA = 0.9
LR_A = 0.001
LR_C = 0.01

env=gym.make('Pendulum-v1')
#env.seed(1)
env.unwrapped

N_S=env.observation_space.shape[0]
A_BOUND=env.action_space.high

actor=Actor(n_features=N_S,lr=LR_A,action_bound=[float(-A_BOUND),float(A_BOUND)])
critic=Critic(n_features=N_S,lr=LR_C)

for i_episode in range(MAX_EPISODE):
    s=env.reset()
    t=0
    ep_rs=[]
    while True:
        if RENDER:env.render()
        a=actor.choose_action(s)

        s_,r,done,info,_=env.step(a)
        r/=10

        td_error=critic.learn(s,r,s_)
        actor.learn(s,a,td_error)

        s=s_
        t+=1
        ep_rs.append(r)
        if t>MAX_EP_STEPS:
            ep_rs_sum=sum(ep_rs)
            if 'running_reward' not in globals():
                running_reward=ep_rs_sum
            else:
                running_reward=running_reward*0.9+ep_rs_sum*0.1
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True
            print('episode: ',i_episode,' reward:',int(running_reward))
            break

