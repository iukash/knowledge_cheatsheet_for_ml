import gym
import numpy as np


class Qlearning:
    """
    Алгоритм Q-learning

    Переменные:
     - self.ganna - коэффициент дисконтирования
     - self.alpha - шаг обучения
     - self.n_episode - количество итераций обучения
     - self.n_env_trajectory - количество траекторий
     - self.mean_total_rewards - средняя награда на каждой итерации
     - self.n_episode_discount - количество итераций до уменьшения шума до нуля

    Функции:
     - self.fit - запуск процесса обучения
     - self.get_action_epsilon_greedy - получение действие при эпсилон жадной стратегии
    """

    def __init__(self, env, gamma, alpha, n_episode, n_episode_discount):
        self.env = env
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.q = np.zeros((self.n_states, self.n_actions))
        self.gamma = gamma
        self.alpha = alpha
        self.n_episode = n_episode
        self.eps_greedy = 1
        self.n_env_trajectory = 0
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
            self.eps_greedy = 1 - episode / self.n_episode_discount
            if self.eps_greedy < 0:
                self.eps_greedy = 0

            total_reward = 0
            state = self.env.reset()
            for _ in range(max_len):
                action = self.get_action_epsilon_greedy(self.q[state])
                state_next, reward, done, _ = self.env.step(action)
                total_reward += reward

                # формула обновления функции Q (обучение)
                self.q[state][action] += self.alpha * (reward + self.gamma * np.max(self.q[state_next]) - self.q[state][action])
                state = state_next

                if done:
                    break

            self.mean_total_rewards.append(np.round(total_reward, 3))


if __name__ == '__main__':
    q_learning = Qlearning(gym.make("Taxi-v3"),0.99, 0.8, 8000, 400)
    q_learning.fit()
    print(f'mean_total_rewards {q_learning.mean_total_rewards}')
