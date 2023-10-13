import gym
import time
import numpy as np


class AgentCrossEntropyLaplace():
    def __init__(self, action_n, taxi_loc_n, pass_loc_n, dest_loc_n, l_laplace):
        self.action_n = action_n
        self.taxi_loc_n = taxi_loc_n
        self.pass_loc_n = pass_loc_n
        self.dest_loc_n = dest_loc_n
        self.l_laplace = l_laplace
        self.model = np.ones(taxi_loc_n*pass_loc_n*dest_loc_n*action_n).reshape(taxi_loc_n, pass_loc_n, dest_loc_n, action_n) / action_n

    def get_action(self, taxi_loc, pass_loc, dest_loc):
        # state: taxi_loc, pass_loc, dest_loc
        try:
            action = np.random.choice(np.arange(self.action_n), p=self.model[taxi_loc][pass_loc][dest_loc])
        except:
            print(self.model[taxi_loc][pass_loc][dest_loc])
            raise Exception('Except method get_action')
        return action

    def fit(self, elite_trajectories):
        new_model = (np.zeros(self.taxi_loc_n*self.pass_loc_n*self.dest_loc_n*self.action_n).
                     reshape(self.taxi_loc_n, self.pass_loc_n, self.dest_loc_n, self.action_n))
        for trajectory in elite_trajectories:
            for taxi_loc, pass_loc, dest_loc, action in zip(trajectory['taxi_loc'], trajectory['pass_loc'],
                                                            trajectory['dest_loc'], trajectory['action']):
                new_model[taxi_loc][pass_loc][dest_loc][action] += 1

        for taxi_loc in range(self.taxi_loc_n):
            for pass_loc in range(self.pass_loc_n):
                for dest_loc in range(self.dest_loc_n):
                    if np.sum(new_model[taxi_loc][pass_loc][dest_loc]) > 0:
                        new_model[taxi_loc][pass_loc][dest_loc] = ((new_model[taxi_loc][pass_loc][dest_loc] + self.l_laplace) /
                                                                   (np.sum(new_model[taxi_loc][pass_loc][dest_loc]) + self.l_laplace*self.action_n))
                    else:
                        new_model[taxi_loc][pass_loc][dest_loc]= self.model[taxi_loc][pass_loc][dest_loc].copy()

        self.model = new_model
        return None


def get_loc_taxi(taxi_row, taxi_col):
    return int(5*taxi_row + taxi_col)


def get_trajectories(env, agent, n_trajectories, vizualize=False, max_len=1000):
    count_done = 0
    trajectories = []
    for _ in range(n_trajectories):
        trajectory = {'taxi_loc': [], 'pass_loc': [], 'dest_loc': [], 'action': [], 'total_reward': float}
        trajectory['total_reward'] = 0
        obs = env.reset(return_info=False)
        for j in range(max_len):
            taxi_row, taxi_col, pass_loc, dest_idx = env.decode(obs)
            trajectory['taxi_loc'].append(get_loc_taxi(taxi_row, taxi_col))
            trajectory['pass_loc'].append(pass_loc)
            trajectory['dest_loc'].append(dest_idx)
            action = agent.get_action(get_loc_taxi(taxi_row, taxi_col), pass_loc, dest_idx)
            obs, reward, done, _ = env.step(action)
            trajectory['action'].append(action)
            trajectory['total_reward'] += reward

            if vizualize:
                env.render()
                time.sleep(0.1)

            if done:
                count_done += 1
                break

        trajectories.append(trajectory)
    return trajectories, count_done


def main_func(env, agent, q, n_iteration, n_trajectories, pr=False):
    # основной цикл
    mean_rewards = []
    for i in range(n_iteration):
        # получение траекторий
        trajectories, count_done = get_trajectories(env, agent, n_trajectories, False, 199)
        # получение вектора наград для расчета квантиля
        rewards = [trajectory['total_reward'] for trajectory in trajectories]
        mean_rewards.append(np.mean(rewards))
        q_quantile = np.quantile(rewards, q)
        if pr:
            print(f'На шаге {i} средняя награда {np.mean(rewards)} из {n_trajectories} завершено {count_done}')
        # получение элитных траекторий
        elite_trajectories = []
        for trajectory in trajectories:
            if trajectory['total_reward'] > q_quantile:
                elite_trajectories.append(trajectory)

        # обучение агента
        agent.fit(elite_trajectories)

    return mean_rewards


#env = gym.make('Taxi-v3')
# actions 0: move south 1: move north 2: move east 3: move west : pickup passenger 5: drop off passenger
#action_n = 6
# location taxi 0 - 24
#taxi_loc_n = 25
# pass_loc 0: R(ed) 1: G(reen) 2: Y(ellow) 3: B(lue) 4: in taxi
#pass_loc_n = 5
# 0: R(ed) 1: G(reen) 2: Y(ellow) 3: B(lue)
#dest_loc_n = 4
#agent = AgentCrossEntropyLaplace(action_n, taxi_loc_n, pass_loc_n, dest_loc_n, l_laplace=0.5)
#main_func(env, agent, 0.5, 20, 1000, True)

