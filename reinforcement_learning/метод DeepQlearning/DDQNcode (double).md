```python
import copy
from collections import deque
import gym
import numpy as np
import torch
import torch.nn as nn
import random


class DQN_DDQN(nn.Module):
    """
    Алгоритм DeepQNetwork Double DQN

    Переменные:
    - self.env - среда
    - self.n_episode - количество итераций обучения
    - self.batch_size - размер выборки для обучения
    - self.gamma - коэффициент дисконтирования
    - self.epsilon_decrease - величина убывания эпсилон
    - self.epsilon_min - минимальное значение эпсилон
    - self.n_actions - количество действий
    - self.mean_total_rewards - сохранение наград для графиков
    - self.n_env_trajectory - количество обращений к среде (траекторий)
    - self.memory - память хранящая четверки state, action, reward, next_state для обучения делая выборки
    - self.network - сеть
    - self.optimizer - оптимизатор
    - self.n_neurons - количество нейронов в однослойной сети
    - self.lr - шаг обучения

    Функции:
     - self.fit - запуск процесса обучения
     - self.policy_improvement - обновление весов нейросети используя часть сохраненных четверок
     - get_action - получение действия
    """

    def __init__(self, env, n_episode, n_neurons, lr=0.001, gamma=0.99, batch_size=64,
                 eps_end=0.001, eps_decay=0.995, tau=0.1):
        super().__init__()
        self.env = env
        self.n_episode = n_episode
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = 1
        self.eps_decay = eps_decay
        self.eps_end = eps_end
        self.n_actions = env.action_space.n
        self.mean_total_rewards = []
        self.n_env_trajectory = 0
        self.tau = tau
        self.memory = deque(maxlen=100000)
        self.network = nn.Sequential(nn.Linear(env.observation_space.shape[0], n_neurons),
                                     nn.ReLU(),
                                     nn.Linear(n_neurons, n_neurons),
                                     nn.ReLU(),
                                     nn.Linear(n_neurons, self.n_actions))
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        self.network_ = copy.deepcopy(self.network)

    def get_action(self, state):
        q_values = self.network(torch.FloatTensor(state))
        argmax_action = torch.argmax(q_values)
        probs = self.epsilon * np.ones(self.n_actions) / self.n_actions
        probs[argmax_action] += 1 - self.epsilon
        action = np.random.choice(np.arange(self.n_actions), p=probs)
        return action

    def policy_improvement(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, int(done), next_state])

        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)

            states, actions, rewards, dones, next_states = map(np.array, list(zip(*batch)))
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            dones = torch.LongTensor(dones)
            next_states = torch.FloatTensor(next_states)

            targets = (rewards + self.gamma * (1 - dones) *
                       self.network_(next_states).gather(1, torch.argmax(self.network(next_states), dim=1).unsqueeze(1)).squeeze(1))
            q_values = self.network(states)[torch.arange(self.batch_size), actions]

            loss = torch.mean((q_values - targets.detach()) ** 2)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.soft_update()

    def soft_update(self):
        for param_network_, param_network in zip(self.network_.parameters(), self.network.parameters()):
            param_network_.data.copy_(self.tau * param_network.data + (1.0 - self.tau) * param_network_.data)

    def fit(self, max_len=500):
        for episode in range(self.n_episode):
            self.n_env_trajectory += 1
            total_reward = 0
            state = self.env.reset()
            for _ in range(max_len):
                action = self.get_action(state)
                state_next, reward, done, _ = self.env.step(action)
                total_reward += reward

                self.policy_improvement(state, action, reward, done, state_next)
                state = state_next

                if done:
                    break

            self.epsilon = max(self.eps_end, self.eps_decay * self.epsilon)
            #print(f'iteration {episode} reward {np.round(total_reward, 3)} epsilon {np.round(self.epsilon, 3)}')
            self.mean_total_rewards.append(np.round(total_reward, 3))


if __name__ == '__main__':
    dqn = DQN_DDQN(env=gym.make('LunarLander-v2'), lr=0.001, n_episode=1000, n_neurons=128, eps_decay=0.99)
    dqn.fit()
    print(f'mean_total_rewards {dqn.mean_total_rewards}')

```