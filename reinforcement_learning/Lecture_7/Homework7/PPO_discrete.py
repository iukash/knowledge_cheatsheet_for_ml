import gym
import numpy as np
import torch
import torch.nn as nn


class PPO_discrete(nn.Module):
    """
    Алгоритм Proximal Policy Optimization

    Переменные:
    - self.env - среда
    - self.n_episode - количество итераций обучения
    - self.n_trajectory  - количество траекторий до обучения агента
    - self.epoch_n - количество шагов в обучении агента
    - self.batch_size - размер выборки для обучения
    - self.gamma - коэффициент дисконтирования
    - self.epsilon - коэффициент ограничивающий частное прогнозов новой и старой политики
    - self.mean_total_rewards - сохранение наград для графиков
    - self.n_env_trajectory - количество обращений к среде (траекторий)
    - self.n_neurons - количество нейронов
    - self.pi_model - нейросеть политики
    - self.v_model - нейросеть прогноза ценности состояния
    - self.pi_lrб self.v_lr - шаги обучения
    - self.pi_optimizer, self.v_optimizer - оптимизаторы

    Функции:
     - self.fit - запуск процесса обучения
     - self.policy_improvement - обучение агента
     - get_action - получение действия
    """

    def __init__(self, env, max_len_trajectory=200, n_episode=50, n_trajectory=20, n_neurons=128, gamma=0.9, batch_size=128,
                 epsilon=0.2, epoch_n=30, pi_lr=0.0005, v_lr=0.0005, is_print=True, eps_decay=0.9):
        super().__init__()
        self.env = env
        self.n_episode = n_episode
        self.n_trajectory = n_trajectory
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = env.action_space.n
        self.epoch_n = epoch_n
        self.mean_total_rewards = []
        self.n_env_trajectory = 0
        self.pi_model = nn.Sequential(nn.Linear(env.observation_space.shape[0], n_neurons), nn.ReLU(),
                                      nn.Linear(n_neurons, self.n_actions))

        self.v_model = nn.Sequential(nn.Linear(env.observation_space.shape[0], n_neurons), nn.ReLU(),
                                     nn.Linear(n_neurons, 1))

        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters(), lr=pi_lr)
        self.v_optimizer = torch.optim.Adam(self.v_model.parameters(), lr=v_lr)
        self.max_len_trajectory = max_len_trajectory
        self.is_print = is_print

    def get_action(self, state):
        probs = torch.nn.functional.softmax(self.pi_model(torch.FloatTensor(state)), dim=0)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return np.array(action)

    def policy_improvement(self, states, actions, rewards, dones):
        states, actions, rewards, dones = map(np.array, [states, actions, rewards, dones])
        rewards, dones = rewards.reshape(-1, 1), dones.reshape(-1, 1)

        returns = np.zeros(rewards.shape)
        returns[-1] = rewards[-1]
        for t in range(returns.shape[0] - 2, -1, -1):
            returns[t] = rewards[t] + (1 - dones[t]) * self.gamma * returns[t + 1]

        states, actions, returns = map(torch.FloatTensor, [states, actions, returns])

        probs = torch.nn.functional.softmax(self.pi_model(torch.FloatTensor(states)), dim=0)
        dist = torch.distributions.Categorical(probs)
        old_log_probs = dist.log_prob(actions).detach()

        for epoch in range(self.epoch_n):

            idxs = np.random.permutation(returns.shape[0])
            for i in range(0, returns.shape[0], self.batch_size):
                b_idxs = idxs[i: i + self.batch_size]
                b_states = states[b_idxs]
                b_actions = actions[b_idxs]
                b_returns = returns[b_idxs]
                b_old_log_probs = old_log_probs[b_idxs]

                b_advantage = b_returns.detach() - self.v_model(b_states)

                b_probs = torch.nn.functional.softmax(self.pi_model(torch.FloatTensor(b_states)), dim=0)
                b_dist = torch.distributions.Categorical(b_probs)
                b_new_log_probs = b_dist.log_prob(b_actions)

                b_ratio = torch.exp(b_new_log_probs - b_old_log_probs)
                pi_loss_1 = b_ratio * b_advantage.detach()
                pi_loss_2 = torch.clamp(b_ratio, 1. - self.epsilon, 1. + self.epsilon) * b_advantage.detach()
                pi_loss = - torch.mean(torch.min(pi_loss_1, pi_loss_2))

                pi_loss.backward()
                self.pi_optimizer.step()
                self.pi_optimizer.zero_grad()

                v_loss = torch.mean(b_advantage ** 2)

                v_loss.backward()
                self.v_optimizer.step()
                self.v_optimizer.zero_grad()

    def fit(self):
        for episode in range(self.n_episode):
            states, actions, rewards, dones = [], [], [], []
            for _ in range(self.n_trajectory):
                self.n_env_trajectory += 1
                total_reward = 0
                state = self.env.reset()
                for _ in range(self.max_len_trajectory):
                    states.append(state)
                    action = self.get_action(state)
                    actions.append(action)
                    state_next, reward, done, _ = self.env.step(action)
                    rewards.append(reward)
                    dones.append(done)
                    total_reward += reward
                    state = state_next

                    if done:
                        break

            self.policy_improvement(states, actions, rewards, dones)

            if self.is_print:
                print(f'iteration {episode} reward {np.round(total_reward, 3)}')
            self.mean_total_rewards.append(np.round(total_reward, 3))


if __name__ == '__main__':
    ppo = PPO_discrete(env=gym.make('CartPole'), max_len_trajectory=500, n_episode=5000,
                           n_trajectory=5, n_neurons=64, gamma=0.99, batch_size=250, epoch_n=30,
                           pi_lr=0.0005, v_lr=0.0005, is_print=True, eps_decay=0.99)
    ppo.fit()
    print(f'mean_total_rewards {ppo.mean_total_rewards}')
