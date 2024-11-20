```python
import gym
import numpy as np
import torch
import torch.nn as nn


class PPO_continuous(nn.Module):
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

    def __init__(self, env, max_len_trajectory=200, n_episode=50, n_trajectory=100, n_neurons=128, gamma=0.99, batch_size=128,
                 epsilon=0.2, epoch_n=30, pi_lr=0.0005, v_lr=0.0005, is_print=True, transform_action=1):
        super().__init__()
        self.env = env
        self.n_episode = n_episode
        self.n_trajectory = n_trajectory
        self.batch_size = batch_size
        self.gamma = gamma
        self.transform_action = transform_action
        self.epsilon = epsilon
        self.epoch_n = epoch_n
        self.mean_total_rewards = []
        self.n_env_trajectory = 0
        self.n_actions = env.action_space.shape[0]
        self.pi_model = nn.Sequential(nn.Linear(env.observation_space.shape[0], n_neurons), nn.ReLU(),
                                      nn.Linear(n_neurons, n_neurons), nn.ReLU(),
                                      nn.Linear(n_neurons, 2 * self.n_actions), nn.Tanh())

        self.v_model = nn.Sequential(nn.Linear(env.observation_space.shape[0], n_neurons), nn.ReLU(),
                                     nn.Linear(n_neurons, n_neurons), nn.ReLU(),
                                     nn.Linear(n_neurons, 1))

        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters(), lr=pi_lr)
        self.v_optimizer = torch.optim.Adam(self.v_model.parameters(), lr=v_lr)
        self.max_len_trajectory = max_len_trajectory
        self.is_print = is_print
        self.mean_pi_losses = []
        self.mean_v_losses = []

    def get_action(self, state):
        actions = []
        output = self.pi_model(torch.FloatTensor(state))
        for i in range(self.n_actions):
            mean = output[2*i]
            log_std = output[2*i+1]
            actions.append(torch.distributions.Normal(mean, torch.exp(log_std)).sample())
        return actions

    def policy_improvement(self, states, actions, rewards, dones, next_states):
        states, actions, rewards, dones, next_states = map(np.array, [states, actions, rewards, dones, next_states])
        rewards, dones = rewards.reshape(-1, 1), dones.reshape(-1, 1)

        states, actions, rewards, next_states = map(torch.FloatTensor, [states, actions, rewards, next_states])

        outputs_old = self.pi_model(states).T
        old_log_probs = torch.FloatTensor()
        for i in range(self.n_actions):
            mean = outputs_old[2*i]
            log_std = outputs_old[2 * i + 1]
            mean, log_std = mean.unsqueeze(1), log_std.unsqueeze(1)
            dist = torch.distributions.Normal(mean, torch.exp(log_std))
            old_log_probs = torch.cat([old_log_probs, dist.log_prob(actions.T[i].reshape(actions.T[i].shape[0], 1))], 1).detach()

        for epoch in range(self.epoch_n):
            pi_losses = []
            v_losses = []
            idxs = np.random.permutation(rewards.shape[0])
            for i in range(0, rewards.shape[0], self.batch_size):
                b_idxs = idxs[i: i + self.batch_size]
                b_states = states[b_idxs]
                b_actions = actions[b_idxs]
                b_rewards = rewards[b_idxs]
                b_next_states = next_states[b_idxs]
                b_old_log_probs = torch.index_select(old_log_probs, 0, torch.LongTensor(b_idxs))

                b_advantage = b_rewards + self.gamma*self.v_model(b_next_states) - self.v_model(b_states)

                outputs_new = self.pi_model(b_states).T
                b_new_log_probs = torch.FloatTensor()
                for i in range(self.n_actions):
                    mean = outputs_new[2 * i]
                    log_std = outputs_new[2 * i + 1]
                    mean, log_std = mean.unsqueeze(1), log_std.unsqueeze(1)
                    dist = torch.distributions.Normal(mean, torch.exp(log_std))
                    b_new_log_probs = torch.cat([b_new_log_probs, dist.log_prob(b_actions.T[i].reshape(b_actions.T[i].shape[0], 1))], 1)

                b_ratio = torch.exp(b_new_log_probs - b_old_log_probs)
                pi_loss_1 = b_ratio * b_advantage.detach()
                pi_loss_2 = torch.clamp(b_ratio, 1. - self.epsilon, 1. + self.epsilon) * b_advantage.detach()
                pi_loss = - torch.mean(torch.min(pi_loss_1, pi_loss_2))
                pi_losses.append(pi_loss.data.numpy())

                pi_loss.backward()
                self.pi_optimizer.step()
                self.pi_optimizer.zero_grad()

                v_loss = torch.mean(b_advantage ** 2)
                v_losses.append(v_loss.data.numpy())

                v_loss.backward()
                self.v_optimizer.step()
                self.v_optimizer.zero_grad()

            self.mean_pi_losses.append(np.mean(pi_losses))
            self.mean_v_losses.append(np.mean(v_losses))

    def fit(self):
        for episode in range(self.n_episode):
            if episode >= self.n_episode - 10:
                self.zero_std = True

            states, actions, rewards, dones, next_states = [], [], [], [], []
            for _ in range(self.n_trajectory):
                self.n_env_trajectory += 1
                total_reward = 0
                state = self.env.reset()
                for _ in range(self.max_len_trajectory):
                    states.append(state)
                    action = self.get_action(state)
                    actions.append(action)
                    state_next, reward, done, _ = self.env.step(self.transform_action*action)
                    rewards.append(reward)
                    dones.append(done)
                    next_states.append(state_next)
                    total_reward += reward
                    state = state_next

                    if done:
                        break

            self.policy_improvement(states, actions, rewards, dones, next_states)
            if self.is_print:
                print(f'iteration {episode} reward {np.round(total_reward, 3)} '
                      f'pi_loss {np.mean(self.mean_pi_losses[episode * self.epoch_n:(episode + 1) * self.epoch_n])} '
                      f'v_loss {np.mean(self.mean_v_losses[episode * self.epoch_n:(episode + 1) * self.epoch_n])}')
            self.mean_total_rewards.append(np.round(total_reward, 3))


if __name__ == '__main__':
    ppo_hw2 = PPO_continuous(env=gym.make('LunarLander-v2', continuous=True), max_len_trajectory=1000, n_episode=500,
                      n_trajectory=50, n_neurons=256, epoch_n=5)
    ppo_hw2.fit()
    print(f'mean_total_rewards {ppo_hw2.mean_total_rewards}')
    print(f'mean_pi_losses{ppo_hw2.mean_pi_losses}')
    print(f'mean_v_losses{ppo_hw2.mean_v_losses}')

```