import numpy as np

from maze_env import Maze
from RL_brain import DeepQNetwork


def update():
    step=0
    for episode in range(1):
        print("episode:{}".format(episode))
        observation = np.array(env.reset())
       # action = RL.choose_action(str(observation))
        while True:
            print("step:{}".format(step))
            env.render()
            action = RL.choose_action(observation)
            observation_, reward, done = env.step(action)
            #action_ = RL.choose_action(str(observation_))
            # RL.learn(str(observation), action, reward, str(observation_),action_) #Sarsa
            #RL.learn(str(observation), action, reward, str(observation_)) #Q-learning
            RL.store_transition(observation,action,reward,observation_)
            if (step>200) and(step%5==0):
                RL.learn()
            #observation = observation_
            #action = action_
            if done:
                break;
            step+=1
    print('Game Over')
    env.destroy()


if __name__ == '__main__':
    env = Maze()
    RL = DeepQNetwork(env.n_actions,env.n_features,learning_rate=0.01,reward_decay=0.9,e_greedy=0.8,replace_target_iter=200,memory_size=2000)

    env.after(20, update)
    env.mainloop()
    RL.plot_cost()