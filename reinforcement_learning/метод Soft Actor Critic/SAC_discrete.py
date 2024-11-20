import gym
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import random
from collections import deque
from torch.nn import functional as F


class replayBuffer:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer = []
        self._next_idx = 0

    def add(self, item):
        if len(self.buffer) > self._next_idx:
            self.buffer[self._next_idx] = item
        else:
            self.buffer.append(item)
        if self._next_idx == self.buffer_size - 1:
            self._next_idx = 0
        else:
            self._next_idx = self._next_idx + 1

    def sample(self, batch_size):
        indices = [random.randint(0, len(self.buffer) - 1) for _ in range(batch_size)]
        states = [self.buffer[i][0] for i in indices]
        actions = [self.buffer[i][1] for i in indices]
        rewards = [self.buffer[i][2] for i in indices]
        n_states = [self.buffer[i][3] for i in indices]
        dones = [self.buffer[i][4] for i in indices]
        return states, actions, rewards, n_states, dones

    def length(self):
        return len(self.buffer)


class QNet(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()

        self.hidden = nn.Linear(4, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, s):
        outs = self.hidden(s)
        outs = F.relu(outs)
        outs = self.output(outs)
        return outs


class SAC_discrete(nn.Module):
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

    def __init__(self, env, max_len_trajectory=200, n_episode=50, n_neurons=128, gamma=0.99, batch_size=64, alpha=1e-3,
                 tau=1e-2, q_lr=1e-3, pi_lr=0.0005, is_print=True, len_deque=100000):
        super().__init__()
        self.env = env
        self.n_episode = n_episode
        self.batch_size = batch_size
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.memory = replayBuffer(len_deque)
        self.mean_total_rewards = []
        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.shape[0]

        self.pi_model = QNet(n_neurons)

        self.q1_model = QNet(n_neurons)
        self.q2_model = QNet(n_neurons)
        self.q1_target_model = QNet(n_neurons)
        self.q2_target_model = QNet(n_neurons)

        self.pi_optimizer = torch.optim.AdamW(self.pi_model.parameters(), pi_lr)
        self.q1_optimizer = torch.optim.AdamW(self.q1_model.parameters(), q_lr)
        self.q2_optimizer = torch.optim.AdamW(self.q2_model.parameters(), q_lr)

        self.max_len_trajectory = max_len_trajectory
        self.is_print = is_print
        self.mean_pi_losses = []

    def predict_actions(self, states):
        logits = self.pi_model(states)
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs)
        return probs, log_probs

    def get_action(self, state):
        with torch.no_grad():
            logits = self.pi_model(torch.tensor(np.expand_dims(state, axis=0))).squeeze(dim=0)
            probs = F.softmax(logits, dim=-1)
            try:
                action = torch.multinomial(probs, num_samples=1).squeeze(dim=0)
            except:
                action = np.argmax(np.round(probs, 0))
        return action.tolist()

    def policy_improvement(self, state, action, reward, done, next_state):
        self.memory.add([state.tolist(), action, reward, next_state.tolist(), float(done)])
        if self.memory.length() > self.batch_size:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            states = torch.tensor(states, dtype=torch.float)
            actions = torch.tensor(actions, dtype=torch.int64)
            rewards = torch.tensor(rewards, dtype=torch.float)
            rewards = rewards.unsqueeze(dim=1)
            next_states = torch.tensor(next_states, dtype=torch.float)
            dones = torch.tensor(dones, dtype=torch.float)
            dones = dones.unsqueeze(dim=1)


            with torch.no_grad():
                q1_tgt_next = self.q1_target_model(next_states)
                q2_tgt_next = self.q2_target_model(next_states)
                next_probs, next_log_probs = self.predict_actions(next_states)
                q1_target = q1_tgt_next.unsqueeze(dim=1) @ next_probs.unsqueeze(dim=2)
                q1_target = q1_target.squeeze(dim=1)
                q2_target = q2_tgt_next.unsqueeze(dim=1) @ next_probs.unsqueeze(dim=2)
                q2_target = q2_target.squeeze(dim=1)
                q_target_min = torch.minimum(q1_target, q2_target)
                h = next_probs.unsqueeze(dim=1) @ next_log_probs.unsqueeze(dim=2)
                h = h.squeeze(dim=1)
                h = -self.alpha * h
                term2 = rewards + self.gamma * (1.0 - dones) * (q_target_min + h)

            self.q1_optimizer.zero_grad()
            one_hot_actions = F.one_hot(actions, num_classes=2).float()
            q_value1 = self.q1_model(states)
            term1 = q_value1.unsqueeze(dim=1) @ one_hot_actions.unsqueeze(dim=2)
            term1 = term1.squeeze(dim=1)
            loss_q1 = F.mse_loss(term1, term2, reduction="none")
            loss_q1.sum().backward()
            self.q1_optimizer.step()

            self.q2_optimizer.zero_grad()
            one_hot_actions = F.one_hot(actions, num_classes=2).float()
            q_value2 = self.q2_model(states)
            term1 = q_value2.unsqueeze(dim=1) @ one_hot_actions.unsqueeze(dim=2)
            term1 = term1.squeeze(dim=1)
            loss_q2 = F.mse_loss(term1, term2, reduction="none")
            loss_q2.sum().backward()
            self.q2_optimizer.step()

            for p in self.q1_model.parameters():
                p.requires_grad = False

            self.pi_optimizer.zero_grad()
            probs, log_probs = self.predict_actions(states)
            q_value = self.q1_model(states)
            term1 = probs
            term2 = q_value - self.alpha * log_probs
            expectation = term1.unsqueeze(dim=1) @ term2.unsqueeze(dim=2)
            expectation = expectation.squeeze(dim=1)
            (-expectation).sum().backward()
            self.pi_optimizer.step()

            for p in self.q1_model.parameters():
                p.requires_grad = True

            self.update_target()

    def update_target(self):
        for var, var_target in zip(self.q1_model.parameters(), self.q1_target_model.parameters()):
            var_target.data = self.tau * var.data + (1.0 - self.tau) * var_target.data
        for var, var_target in zip(self.q2_model.parameters(), self.q2_target_model.parameters()):
            var_target.data = self.tau * var.data + (1.0 - self.tau) * var_target.data

    def fit(self):
        for episode in range(self.n_episode):
            total_reward = 0
            state = self.env.reset()
            for _ in range(self.max_len_trajectory):
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward

                self.policy_improvement(state, action, reward, done, next_state)
                state = next_state

                if done:
                    break

            if self.is_print:
                print(f'iteration {episode} reward {np.round(total_reward, 3)}')
            self.mean_total_rewards.append(np.round(total_reward, 3))


if __name__ == '__main__':
    sac = SAC_discrete(env=gym.make('CartPole'), max_len_trajectory=500, n_episode=1000, n_neurons=64, gamma=0.99,
                        batch_size=250, alpha=0.1, tau=0.002, q_lr=0.0005, pi_lr=0.0005, is_print=True, len_deque=20000)
    sac.fit()
    print(f'mean_total_rewards {sac.mean_total_rewards}')
