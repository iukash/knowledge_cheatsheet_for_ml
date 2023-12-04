import gym
import numpy as np
import torch
import torch.nn as nn


class DeepCrossEntropy(nn.Module):
    """
    Алгоритм DeepCrossEntropy

    Переменные:
     - self.q - квантиль для элитных траекторий ((1 - q)*100 = процент элитных траекторий)
     - self.n_env_trajectory - количество траекторий
     - self.is_print - печатать информацию на каждой итерации
     - self.n_episode - количество итераций
     - self.eps - коэффициент шума
     - self.n_neurons - количество нейронов в однослойной сети
     - self.lr - шаг обучения

    Функции:
     - self.fit - запуск процесса обучения
     - self.get_trajectories - получение траекторий
     - self.policy_improvement - улучшение политики через элитные траектории
     - get_action - получение действия
    """

    def __init__(self, env, q, n_trajectories, n_episode, n_neurons, eps_discount, lr, is_print=True, init_model=None):
        super().__init__()
        self.env = env
        self.q = q
        self.n_trajectories = n_trajectories
        self.n_episode = n_episode
        self.is_print = is_print
        self.eps = 1.
        self.eps_discount = eps_discount
        self.n_actions = env.action_space.n
        self.mean_total_rewards = []
        self.n_env_trajectory = 0
        self.network = nn.Sequential(nn.Linear(env.observation_space.shape[0], n_neurons),
                                     nn.ReLU(),
                                     nn.Linear(n_neurons, self.n_actions))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.softmax = nn.Softmax(dim=0)
        self.loss = nn.CrossEntropyLoss()
        if init_model is None:
            self.uniform_policy = np.arange(self.n_actions) / self.n_actions
        else:
            self.uniform_policy = init_model

    def forward(self, input_):
        return self.network(input_)

    def get_action(self, state):
        logits = self.forward(torch.FloatTensor(state))
        probs = (1 - self.eps)*self.softmax(logits).detach().numpy() + self.eps*self.uniform_policy
        probs = probs / np.sum(probs) #нормировка необходимая при введении эпсилон
        output = np.random.choice(self.n_actions, p=probs)
        return output

    def fit(self):
        for i in range(self.n_episode):
            # получение траекторий
            trajectories = self.get_trajectories()
            # получение вектора наград для расчета квантиля
            rewards = [trajectory['total_reward'] for trajectory in trajectories]
            self.mean_total_rewards.append(np.mean(rewards))
            q_quantile = np.quantile(rewards, self.q)
            if self.is_print:
                print(f'На шаге {i} средняя награда {np.mean(rewards)} eps {self.eps}')
            # получение элитных траекторий
            elite_trajectories = []
            for trajectory in trajectories:
                if trajectory['total_reward'] > q_quantile:
                    elite_trajectories.append(trajectory)

            # обучение агента
            if len(elite_trajectories) > 0:
                self.policy_improvement(elite_trajectories)

    def policy_improvement(self, elite_trajectories):
        elite_states = []
        elite_actions = []
        for elite_trajectory in elite_trajectories:
            for state, action in zip(elite_trajectory['state'], elite_trajectory['action']):
                elite_states.append(state)
                elite_actions.append(action)

        elite_states = torch.FloatTensor(np.array(elite_states))
        elite_actions = torch.LongTensor(np.array(elite_actions))
        predict_actions = self.forward(elite_states)
        loss = self.loss(predict_actions, elite_actions)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.eps -= self.eps_discount
        self.eps = np.clip(self.eps, 0, 1.)

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


if __name__ == '__main__':
    dce = DeepCrossEntropy(gym.make("LunarLander-v2"), q=0.5, n_trajectories=100, n_episode=1000, n_neurons=128,
                           eps_discount=0.05, lr=0.01, is_print=True)
    dce.fit()
    print(f'mean_total_rewards {dce.mean_total_rewards}')
