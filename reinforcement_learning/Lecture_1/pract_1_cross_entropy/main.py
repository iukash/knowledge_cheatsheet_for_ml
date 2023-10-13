import gym
import gym_maze
import numpy as np
import time

# создание среды
env = gym.make('maze-sample-5x5-v0')
state_n = 25
action_n = 4


# случайный агент
class RandomAgent():
    def __init__(self, action_n):
        self.action_n = action_n

    def get_action(self, state):
        action = np.random.randint(self.action_n)
        return action


# агент на кросс-энтропии
class AgentCrossEntropy():
    def __init__(self, state_n, action_n):
        self.state_n = state_n
        self.action_n = action_n
        self.model = np.ones((state_n, action_n)) / action_n

    def get_action(self, state):
        action = np.random.choice(np.arange(self.action_n), p=self.model[state])
        return int(action)

    def fit(self, elite_trajectory):
        new_model = np.zeros((state_n, action_n))
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_model[state][action] += 1

        for state in range(self.state_n):
            if np.sum(new_model[state]) > 0:
                new_model[state] /= np.sum(new_model[state])
            else:
                new_model[state] = self.model[state].copy()

        self.model = new_model
        return None

# функция перевода состояния из двузначного числа в однозначное
def get_state(obs):
    return int(np.sqrt(state_n)*obs[0] + obs[1])


# получить траекторию
def get_trajectory(env, agent, max_len=1000, vizualize=False):
    trajectory = {'states': [], 'actions': [], 'rewards': []}

    # инициализация начального состояния
    obs = env.reset()
    state = get_state(obs)

    # стратегия действия агента
    for _ in range(max_len):
        trajectory['states'].append(state)

        action = agent.get_action(state)
        trajectory['actions'].append(action)

        next_obs, reward, done, _ = env.step(action)
        trajectory['rewards'].append(reward)

        state = get_state(next_obs)

        if vizualize:
            time.sleep(0.01)
            env.render()

        if done:
            break

    return trajectory


# инициализация агента
agent = AgentCrossEntropy(state_n, action_n)
q_param = 0.9
iteration_n = 100
trajectory_n = 50

for iteration in range(iteration_n):
    # оценка политики
    trajectories = [get_trajectory(env, agent, 100, vizualize=False) for _ in range(trajectory_n)]
    total_reward = [np.sum(trajectory['rewards']) for trajectory in trajectories]
    print('iteration:', iteration, 'mean total reward:', np.mean(total_reward))

    # улучшение политики
    quantile = np.quantile(total_reward, q_param)
    elite_trajectories = []
    for trajectory in trajectories:
        total_reward = np.sum(trajectory['rewards'])
        if total_reward > quantile:
            elite_trajectories.append(trajectory)

    agent.fit(elite_trajectories)
