import gym
import numpy as np


class CrossEntropy:
    """
    Алгоритм Cross_Entropy

    Переменные:
     - self.q - квантиль для элитных траекторий ((1 - q)*100 = процент элитных траекторий)
     - self.l_laplace - сглаживание по Лапласу (если None - нет сглаживания)
     - self.n_env_trajectory - количество траекторий
     - self.is_print - печатать информацию на каждой итерации

    Функции:
     - self.fit - запуск процесса обучения
     - self.get_trajectories - получение траекторий
     - self.policy_improvement - улучшение политики через элитные траектории
    """

    def __init__(self, env, q, n_trajectories, n_episode, is_print=False, l_laplace=None, init_model=None):
        self.env = env
        self.q = q
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.l_laplace = l_laplace
        if init_model is None:
            self.model = np.ones(self.n_states * self.n_actions).reshape(self.n_states, self.n_actions) / self.n_actions
        else:
            self.model = np.dot(np.ones(self.n_states).reshape(self.n_states, 1), init_model.reshape(1, len(init_model)))
        self.mean_total_rewards = []
        self.n_env_trajectory = 0
        self.n_trajectories = n_trajectories
        self.n_episode = n_episode
        self.is_print = is_print

    def get_action(self, state):
        return np.random.choice(range(self.n_actions), p=list(self.model[state]))

    def policy_improvement(self, elite_trajectories):
        new_model = np.zeros(self.n_states * self.n_actions).reshape(self.n_states, self.n_actions)

        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['state'], trajectory['action']):
                new_model[state][action] += 1

        for state in range(self.n_states):
            if np.sum(list(new_model[state])) > 0:
                if self.l_laplace is None:
                    new_model[state] /= np.sum(new_model[state])
                else:
                    new_model[state] = ((new_model[state] + self.l_laplace) / (np.sum(new_model[state]) +
                                                                               self.l_laplace * self.n_actions))
            else:
                new_model[state] = self.model[state].copy()

        self.model = new_model

    def get_trajectories(self, max_len=1000):
        trajectories = []
        for _ in range(self.n_trajectories):
            self.n_env_trajectory += 1
            trajectory = {'state': [], 'action': [], 'total_reward': 0}
            state = self.env.reset()
            for j in range(max_len):
                trajectory['state'].append(state)
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                trajectory['action'].append(action)
                trajectory['total_reward'] += reward
                state = next_state

                if done:
                    break

            trajectories.append(trajectory)
        return trajectories

    def fit(self):
        for i in range(self.n_episode):
            # получение траекторий
            trajectories = self.get_trajectories()
            # получение вектора наград для расчета квантиля
            rewards = [trajectory['total_reward'] for trajectory in trajectories]
            self.mean_total_rewards.append(np.mean(rewards))
            q_quantile = np.quantile(rewards, self.q)
            if self.is_print:
                print(f'На шаге {i} средняя награда {np.mean(rewards)}')
            # получение элитных траекторий
            elite_trajectories = []
            for trajectory in trajectories:
                if trajectory['total_reward'] > q_quantile:
                    elite_trajectories.append(trajectory)

            # обучение агента
            self.policy_improvement(elite_trajectories)


if __name__ == '__main__':
    ce = CrossEntropy(gym.make("Acrobot-v1"), q=0.5, n_trajectories=400, n_episode=15, is_print=True)
                      #init_model=np.array([0.5, 0, 0.5]))
    ce.fit()
    print(f'mean_total_rewards {ce.mean_total_rewards}')
