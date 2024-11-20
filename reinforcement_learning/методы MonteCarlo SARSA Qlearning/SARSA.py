import gym
import numpy as np


class SARSA:
    """
    Алгоритм SARSA

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

    def __init__(self, env, gamma, alpha, n_episode, n_episode_discount):
        self.env = env
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.q = np.zeros((self.n_states, self.n_actions))
        self.n = np.zeros((self.n_states, self.n_actions))
        self.gamma = gamma
        self.alpha = alpha
        self.n_episode = n_episode
        self.eps_greedy = 1
        self.n_env_trajectory = 0
        self.policy = np.ones((self.n_states, self.n_actions)) / self.n_actions
        self.mean_total_rewards = []
        self.n_episode_discount = n_episode_discount

    def get_action_epsilon_greedy(self, q_values):
        max_action = np.argmax(q_values)
        returns = np.ones(self.n_actions) * self.eps_greedy / self.n_actions
        returns[max_action] += 1 - self.eps_greedy
        return np.random.choice(range(self.n_actions), p=returns)

    def fit(self, max_len=1000):
        for episode in range(self.n_episode):
            self.n_env_trajectory += 1
            self.eps_greedy = 1 / (episode + 1)

            total_reward = 0
            state = self.env.reset()
            action = self.get_action_epsilon_greedy(self.q[state])
            for _ in range(max_len):
                state_next, reward, done, _ = self.env.step(action)
                total_reward += reward
                action_next = self.get_action_epsilon_greedy(self.q[state_next])

                # формула обновления функции Q (обучение)
                self.q[state][action] += self.alpha*(reward + self.gamma * self.q[state_next][action_next] - self.q[state][action])

                state = state_next
                action = action_next

                if done:
                    break

            self.mean_total_rewards.append(np.round(total_reward, 3))


if __name__ == '__main__':
    sarsa = SARSA(gym.make('Acrobot-v1'),0.99, 0.5, 10000, 900)
    sarsa.fit()
    print(f'mean_total_rewards {sarsa.mean_total_rewards}')
