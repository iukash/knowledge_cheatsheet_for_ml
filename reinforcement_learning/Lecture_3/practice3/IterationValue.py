from Frozen_Lake import FrozenLakeEnv
import numpy as np


class IterationValue():
    def __init__(self, env, gamma, eps, iter_n):
        self.env = env
        self.gamma = gamma
        self.eps = eps
        self.iter_n = iter_n
        self.v = {}
        self.v_prev = {}
        self.q = {}
        self.policy = {}
        self.n_env = 0

    def create_policy(self):
        self.n_env += 1
        for state in self.env.get_all_states():
            self.n_env += 1
            if len(self.env.get_possible_actions(state)) > 0:
                self.policy[state] = {}
                self.policy[state]['left'] = 0
                self.policy[state]['down'] = 0
                self.policy[state]['right'] = 0
                self.policy[state]['up'] = 0
                self.policy[state][max(self.q[state], key=self.q[state].get)] = 1


    def init_v(self):
        self.n_env += 1
        for state in self.env.get_all_states():
            self.v[state] = 0
            self.v_prev[state] = 0

    def count_v(self):
        self.n_env += 1
        for state in self.env.get_all_states():
            if len(self.q[state]) > 0:
                self.v[state] = max(self.q[state].values())

    def count_q(self):
        self.q = {}
        self.n_env += 1
        for state in self.env.get_all_states():
            self.n_env += 1
            self.q[state] = {}
            for action in self.env.get_possible_actions(state):
                self.n_env += 1
                self.q[state][action] = 0
                next_states = self.env.get_next_states(state, action)
                for next_state in next_states:
                    self.n_env += 2
                    prob_next_state = self.env.get_transition_prob(state, action, next_state)
                    self.q[state][action] += prob_next_state*(self.env.get_reward(state, action, next_state) +
                                                              self.gamma*self.v[next_state])

    def criterion_stop(self):
        self.n_env += 1
        for state in self.env.get_all_states():
            if abs(self.v[state] - self.v_prev[state]) > self.eps:
                return False
        return True

    def fit(self):
        self.init_v()
        for i in range(self.iter_n):
            self.v_prev = self.v.copy()
            self.count_q()
            self.count_v()
            if self.criterion_stop():
                break
        self.create_policy()
        return self.policy

def eval_policy(env, n, policy, max_len = 1000):
    total_rewards = []
    for _ in range(n):
        total_reward = 0
        state = env.reset()
        for _ in range(max_len):
            action = np.random.choice(env.get_possible_actions(state), p=list(policy[state].values()))
            state, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                break

        total_rewards.append(total_reward)

    return np.mean(total_rewards)

if __name__ == '__main__':
    env = FrozenLakeEnv()
    ip = IterationValue(env, 0.9999, 0.001, 100)
    policy = ip.fit()
    print(f'mean_reward {eval_policy(env, 1000, policy)}')
    print(f'n_env {ip.n_env}')
