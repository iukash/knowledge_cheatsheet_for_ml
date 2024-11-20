```python
import gym
import numpy as np


class MonteCarlo:
    """
    Алгоритм Monte-Carlo

    Переменные:
     - self.ganna - коэффициент дисконтирования
     - self.alpha - шаг обучения
     - self.n_episode - количество итераций обучения
     - self.n_env_trajectory - количество траекторий
     - self.mean_total_rewards - средняя награда на каждой итерации

    Функции:
     - self.fit - запуск процесса обучения
     - self.get_action_epsilon_greedy - получение действие при эпсилон жадной стратегии
    """

    def __init__(self, env, gamma, n_episode, eps_version=0):
        self.env = env
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.q = np.zeros((self.n_states, self.n_actions))
        self.n = np.zeros((self.n_states, self.n_actions))
        self.gamma = gamma
        self.n_episode = n_episode
        self.n_env_trajectory = 0
        self.eps = 1
        self.eps_decay = 0.995
        self.policy = np.ones((self.n_states, self.n_actions)) / self.n_actions
        self.mean_total_rewards = []
        self.eps_version = eps_version

    def get_action_epsilon_greedy(self, q_values):
        max_action = np.argmax(q_values)
        returns = np.ones(self.n_actions) * self.eps / self.n_actions
        returns[max_action] += 1 - self.eps
        return np.random.choice(range(self.n_actions), p=returns)

    def fit(self, max_len=1000):
        for episode in range(self.n_episode):
            self.n_env_trajectory += 1
            if self.eps_version == 0:
                self.eps = 1 / (episode + 1)
            elif self.eps_version == 1:
                self.eps = 1 - episode / self.n_episode
            elif self.eps_version == 2:
                self.eps *= self.eps_decay

            state = self.env.reset()
            trajectory = {'states': [], 'actions': [], 'rewards': []}

            # получение траектории
            for _ in range(max_len):
                action = self.get_action_epsilon_greedy(self.q[state])
                state_next, reward, done, _ = self.env.step(action)
                trajectory['states'].append(state)
                trajectory['actions'].append(action)
                trajectory['rewards'].append(reward)
                state = state_next

                if done:
                    break

            # расчет ценности для каждой пары (состояние, действие)
            real_trajectory_len = len(trajectory)
            gt = np.zeros(real_trajectory_len + 1)
            for t in range(real_trajectory_len - 1, -1, -1):
                gt[t] = trajectory['rewards'][t] + self.gamma * gt[t + 1]

            # расчет функции Q для каждой пары (состояние, действие)
            for t in range(real_trajectory_len):
                state = trajectory['states'][t]
                action = trajectory['actions'][t]
                self.q[state][action] += (gt[t] - self.q[state][action]) / (1 + self.n[state][action])
                self.n[state][action] += 1

            self.mean_total_rewards.append(np.round(np.sum(trajectory['rewards']), 3))


if __name__ == '__main__':
    mc = MonteCarlo(gym.make("Taxi-v3"), 0.99, 3000)
    mc.fit()
    print(mc.eps)
    print(f'mean_total_rewards {mc.mean_total_rewards}')


```
