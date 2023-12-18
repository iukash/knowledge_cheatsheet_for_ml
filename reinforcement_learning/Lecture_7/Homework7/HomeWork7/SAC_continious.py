import gym
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import random
from collections import deque


class SAC_continuous(nn.Module):
    """
    Алгоритм Soft Actor Critic

    Переменные:
    - self.env - среда
    - self.n_episode - количество итераций обучения
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

    def __init__(self, env, max_len_trajectory=200, n_episode=50, n_neurons=128, gamma=0.99,
                 batch_size=64, alpha=1e-3, tau=1e-2, q_lr=1e-3, pi_lr=0.0005, is_print=True,
                 transform_action=1):
        super().__init__()
        self.env = env
        self.n_episode = n_episode
        self.batch_size = batch_size
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.memory = deque(maxlen=100000)
        self.mean_total_rewards = []
        self.n_actions = env.action_space.shape[0]
        self.n_states = env.observation_space.shape[0]
        self.transform_action = transform_action

        self.pi_model = nn.Sequential(nn.Linear(self.n_states, n_neurons), nn.ReLU(),
                                      nn.Linear(n_neurons, n_neurons), nn.ReLU(),
                                      nn.Linear(n_neurons, 2 * self.n_actions), nn.Tanh())

        self.q1_model = nn.Sequential(nn.Linear(self.n_states + self.n_actions, n_neurons), nn.ReLU(),
                                      nn.Linear(n_neurons, n_neurons), nn.ReLU(),
                                      nn.Linear(n_neurons, 1))

        self.q2_model = nn.Sequential(nn.Linear(self.n_states + self.n_actions, n_neurons), nn.ReLU(),
                                      nn.Linear(n_neurons, n_neurons), nn.ReLU(),
                                      nn.Linear(n_neurons, 1))

        self.pi_optimizer = torch.optim.Adam(self.pi_model.parameters(), pi_lr)
        self.q1_optimizer = torch.optim.Adam(self.q1_model.parameters(), q_lr)
        self.q2_optimizer = torch.optim.Adam(self.q2_model.parameters(), q_lr)
        self.q1_target_model = deepcopy(self.q1_model)
        self.q2_target_model = deepcopy(self.q2_model)

        self.max_len_trajectory = max_len_trajectory
        self.is_print = is_print
        self.mean_pi_losses = []
        self.mean_v_losses = []

    def predict_actions(self, states):
        means, log_stds = self.pi_model(states).T
        means, log_stds = means.unsqueeze(1), log_stds.unsqueeze(1)
        dists = torch.distributions.Normal(means, torch.exp(log_stds))
        actions = dists.rsample()
        log_probs = dists.log_prob(actions)
        return actions, log_probs

    def get_action(self, state):
        actions = []
        state = torch.FloatTensor(state).unsqueeze(0)
        for i in range(self.n_actions):
            action, _ = self.predict_actions(state)
            actions.append(action.squeeze(1).detach().numpy())
        return np.array(actions)

    def policy_improvement(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, done, next_state])
        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(torch.FloatTensor, zip(*batch))
            rewards, dones = rewards.unsqueeze(1), dones.unsqueeze(1)

            next_actions, next_log_probs = self.predict_actions(next_states)
            next_states_and_actions = torch.concatenate((next_states, next_actions), dim=1)
            next_q1_values = self.q1_target_model(next_states_and_actions)
            next_q2_values = self.q2_target_model(next_states_and_actions)
            next_min_q_values = torch.min(next_q1_values, next_q2_values)
            targets = rewards + self.gamma * (1 - dones) * (next_min_q_values - self.alpha * next_log_probs)

            states_and_actions = torch.concatenate((states, actions), dim=1)
            q1_loss = torch.mean((self.q1_model(states_and_actions) - targets.detach()) ** 2)
            q2_loss = torch.mean((self.q2_model(states_and_actions) - targets.detach()) ** 2)
            self.update_model(q1_loss, self.q1_optimizer, self.q1_model, self.q1_target_model)
            self.update_model(q2_loss, self.q2_optimizer, self.q2_model, self.q2_target_model)

            pred_actions, log_probs = self.predict_actions(states)
            states_and_pred_actions = torch.concatenate((states, pred_actions), dim=1)
            q1_values = self.q1_model(states_and_pred_actions)
            q2_values = self.q2_model(states_and_pred_actions)
            min_q_values = torch.min(q1_values, q2_values)
            pi_loss = - torch.mean(min_q_values - self.alpha * log_probs)
            self.update_model(pi_loss, self.pi_optimizer)
            return pi_loss.detach().numpy()
        return 500

    def update_model(self, loss, optimizer, model=None, target_model=None):
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if model != None and target_model != None:
            for param, terget_param in zip(model.parameters(), target_model.parameters()):
                new_terget_param = (1 - self.tau) * terget_param + self.tau * param
                terget_param.data.copy_(new_terget_param)

    def fit(self):
        for episode in range(self.n_episode):
            total_reward = 0
            state = self.env.reset()
            pi_losses = []
            for _ in range(self.max_len_trajectory):
                action = self.get_action(state)
                action = action.reshape(-1)
                next_state, reward, done, _ = self.env.step(self.transform_action*action)
                total_reward += reward

                pi_losses.append(self.policy_improvement(state, action, reward, done, next_state))
                state = next_state

                if done:
                    break

            if self.is_print:
                print(f'iteration {episode} reward {np.round(total_reward, 3)} ',
                      f'pi_loss {np.mean(pi_losses)}')
                      #f'v_loss {np.mean(self.mean_v_losses[episode * self.epoch_n:(episode + 1) * self.epoch_n])}')
            self.mean_total_rewards.append(np.round(total_reward, 3))


if __name__ == '__main__':
    sac = SAC_continuous(env=gym.make('Pendulum-v1'), transform_action=2, n_episode=100)
    sac.fit()
    print(f'mean_total_rewards {sac.mean_total_rewards}')
    print(f'mean_pi_losses{sac.mean_pi_losses}')
    print(f'mean_v_losses{sac.mean_v_losses}')
