import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD=400
RENDER  = False

env = gym.make('CartPole-v1')
#env.seed(1)
env=env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(n_actions=env.action_space.n,n_features=env.observation_space.shape[0],learning_rate=0.02,reward_decay=0.99)

for i_episode in range(3000):
    observation=env.reset()

    while True:
        if RENDER:env.render()

        action = RL.choose_action(observation)
        observation_,reward,done,info=env.step(action)
        RL.store_transition(observation,action,reward)

        if done:
            ep_re_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_re_sum
            else:
                running_reward = running_reward*0.99+ep_re_sum*0.01

            print("episode:",i_episode," reward:",int(running_reward))

            vt=RL.learn()

            if i_episode==0:
                plt.plot(vt)
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break
        observation = observation_