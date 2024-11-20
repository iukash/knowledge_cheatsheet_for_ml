```python
from Frozen_Lake import FrozenLakeEnv  
import numpy as np  
  
  
class IterationPolicy():  
    def __init__(self, env, gamma, eval_iter_n, iter_n, eps_iter, is_init_v=False):  
        self.env = env  
        self.gamma = gamma  
        self.eval_iter_n = eval_iter_n  
        self.iter_n = iter_n  
        self.is_init_v = is_init_v  
        self.eps_iter = eps_iter  
        self.policy = {}  
        self.v = {}  
        self.q_prev = {}  
        self.q = {}  
        self.n_env = 0  
  
    def init_policy(self):  
        self.n_env += 1  
        for state in self.env.get_all_states():  
            self.policy[state] = {}  
            self.n_env += 1  
            for action in self.env.get_possible_actions(state):  
                self.policy[state][action] = 1 / len(self.env.get_possible_actions(state))  
  
    def init_v(self):  
        self.n_env += 1  
        for state in self.env.get_all_states():  
            self.v[state] = 0  
  
    def init_q_prev(self):  
        self.n_env += 1  
        for state in self.env.get_all_states():  
            self.q[state] = {}  
            self.n_env += 1  
            for action in self.env.get_possible_actions(state):  
                self.q[state][action] = 0  
  
    def count_v(self):  
        self.n_env += 1  
        for state in self.env.get_all_states():  
            self.v[state] = 0  
            self.n_env += 1  
            for action in self.env.get_possible_actions(state):  
                self.v[state] += self.policy[state][action]*self.q[state][action]  
  
    def count_q(self):  
        self.q = {}  
        self.n_env += 1  
        for state in self.env.get_all_states():  
            self.q[state] = {}  
            self.n_env += 1  
            for action in self.env.get_possible_actions(state):  
                self.q[state][action] = 0  
                self.n_env += 1  
                next_states = self.env.get_next_states(state, action)  
                for next_state in next_states:  
                    self.n_env += 2  
                    prob_next_state = self.env.get_transition_prob(state, action, next_state)  
                    self.q[state][action] += prob_next_state*(self.env.get_reward(state, action, next_state) +  
                                                              self.gamma*self.v[next_state])  
  
    def policy_evaluation(self):  
        if self.is_init_v:  
            self.init_v()  
        for i in range(self.eval_iter_n):  
            self.q_prev = self.q.copy()  
            self.count_q()  
            self.count_v()  
            if self.criterion_stop_evaluation():  
                break  
  
    def criterion_stop_evaluation(self):  
        self.n_env += 1  
        for state in self.env.get_all_states():  
            self.n_env += 1  
            for action in self.env.get_possible_actions(state):  
                if abs(self.q[state][action] - self.q_prev[state][action]) > self.eps_iter:  
                    return False  
        return True  
    def policy_improvement(self):  
        self.n_env += 1  
        for state in self.env.get_all_states():  
            self.n_env += 1  
            if len(self.env.get_possible_actions(state)) > 0:  
                self.policy[state]['left'] = 0  
                self.policy[state]['down'] = 0  
                self.policy[state]['right'] = 0  
                self.policy[state]['up'] = 0  
                self.policy[state][max(self.q[state], key=self.q[state].get)] = 1  
  
    def fit(self):  
        self.init_policy()  
        self.init_v()  
        self.init_q_prev()  
        for i in range(self.iter_n):  
            self.policy_evaluation()  
            self.policy_prev = self.policy.copy()  
            self.policy_improvement()  
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
    ip = IterationPolicy(env, 0.9999, 100, 7, 0.0005)  
    policy = ip.fit()  
    print(f'mean_reward {eval_policy(env, 1000, policy)}')  
    print(f'n_env {ip.n_env}')
```