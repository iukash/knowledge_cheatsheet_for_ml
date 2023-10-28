import gym
import time
import torch
import torch.nn as nn
import numpy as np


env = gym.make('Acrobot-v1')

action_n = 3


class RandomAgent():
    def __init__(self, action_n):
        self.action_n = action_n

    def get_action(self, state):
        return np.random.randint(0, self.action_n)

    def fit(self, elite_trajectories):
        pass


class CEMDL(nn.Module):
    def __init__(self, action_n, eps):
        super().__init__()
        self.action_n = action_n
        self.network = nn.Sequential(nn.Linear(6, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 3))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        self.softmax = nn.Softmax(dim=0)
        self.loss = nn.CrossEntropyLoss()
        self.uniform_policy = np.arange(action_n) / action_n
        self.eps = eps

    def forward(self, x):
        return self.network(x)

    def get_action(self, state):
        logits = self.forward(torch.FloatTensor(state))
        print(logits.detach().numpy())
        probs = (1 - self.eps)*self.softmax(logits).detach().numpy() + self.eps*self.uniform_policy
        probs = probs / np.sum(probs) #нормировка необходимая при введении эпсилон
        output = np.random.choice(self.action_n, p=probs)
        return output

    def fit(self, elite_trajectories):
        elite_states = []
        elite_actions = []
        for elite_trajectory in elite_trajectories:
            for state, action in zip(elite_trajectory['states'], elite_trajectory['actions']):
                elite_states.append(state)
                elite_actions.append(action)

        elite_states = torch.FloatTensor(np.array(elite_states))
        elite_actions = torch.LongTensor(np.array(elite_actions))
        predict_actions = self.forward(elite_states)
        loss = self.loss(predict_actions, elite_actions)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.eps = 0


def get_trajectories(env, agent, n_trajectories, visualize=False, max_len=500):
    trajectories = []
    for _ in range(n_trajectories):
        trajectory = {'states': [], 'actions': [], 'total_reward': 0}
        state = env.reset()
        for _ in range(max_len):
            trajectory['states'].append(state)
            action = agent.get_action(state)
            trajectory['actions'].append(action)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            trajectory['total_reward'] += reward

            if done:
                break

            if visualize:
                time.sleep(0.02)
                env.render()
        trajectories.append(trajectory)

    return trajectories

def select_elite_trajectories(trajectories, q):
    elite_trajectories = []
    quantile = np.quantile([trajectory['total_reward'] for trajectory in trajectories], q=q)
    for trajectory in trajectories:
        if trajectory['total_reward'] > quantile:
            elite_trajectories.append(trajectory)
    return elite_trajectories


def main_func(env, agent, n_iteration, n_trajectories, q, visualize=False, is_print=False, max_len_trajectory=500):
    mean_total_reward = []
    for i in range(n_iteration):
        trajectories = get_trajectories(env, agent, n_trajectories, visualize, max_len_trajectory)
        if is_print:
            print(f'step: {i} mean_total_reward: {np.mean([trajectory["total_reward"] for trajectory in trajectories])}')
        mean_total_reward.append(np.mean([trajectory["total_reward"] for trajectory in trajectories]))
        elite_trajectories = select_elite_trajectories(trajectories, q)
        if len(elite_trajectories) > 0:
            agent.fit(elite_trajectories)
    return mean_total_reward

class CEMDL_update(CEMDL):
    def __init__(self, action_n, eps):
        super().__init__(action_n, eps)
        self.uniform_policy = np.array([0.5, 0, 0.5]) #np.arange(action_n) / action_n

agent = CEMDL(action_n, 1.0)
# train
#main_func(env, agent, 20, 20, 0.9, False, True)

# test
main_func(env, agent, 1, 1, 0.9, True)

# поиграть руками
#import keyboard
#def working_keyboard():
#    done = False
#    total_reward = 0
#    state = None
#    while not done:
#        time.sleep(0.02)
#        if keyboard.is_pressed("a"):
#            obs, reward, done, _ = env.step(0)
#        elif keyboard.is_pressed("s"):
#            obs, reward, done, _ = env.step(2)
#        else:
#            obs, reward, done, _ = env.step(1)
#        state = obs
#        #print(state)
#        total_reward += reward
#        env.render()
#
#    print(f'total_reward: {total_reward}')
#
#working_keyboard()
#