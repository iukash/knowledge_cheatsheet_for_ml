import gym
import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from matplotlib import animation
import matplotlib.pyplot as plt
import gym


def save_frames_as_gif(env, frames, filename='MountainCar.gif'):
    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(filename="MountainCar.gif", writer="imagemagick")


class RandomAgent():
    def __init__(self, action_left_range, action_right_range):
        self.action_right_range = action_right_range
        self.action_left_range = action_left_range

    def get_action(self, state):
        action = 2 * np.random.random_sample() - 1
        return [action]

    def fit(self, elite_trajectories):
        pass

class CEMDL(nn.Module):
    def __init__(self, action_left_range, action_right_range, eps, eps_discount, lr):
        super().__init__()
        self.action_left_range = action_left_range
        self.action_right_range = action_right_range
        self.eps = eps
        self.eps_discount = eps_discount
        self.network = nn.Sequential(nn.Linear(2, 64), nn.ReLU(),
                                     nn.Linear(64, 1))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.L1Loss()
        self.tanh = nn.Tanh()

    def forward(self, input_):
        return self.network(input_)

    def get_action(self, state):
        logits = self.forward(torch.FloatTensor(state)).detach().numpy()
        noise = (2 * np.random.random_sample() - 1)*15
        output_network = (1 - self.eps)*logits + self.eps * noise
        #result = np.clip(output_network + self.eps * noise, a_min=self.action_left_range, a_max=self.action_right_range)
        return output_network

    def fit(self, elite_trajectories):
        elite_states = []
        elite_actions = []
        for elite_trajectory in elite_trajectories:
            for state, action in zip(elite_trajectory['states'], elite_trajectory['actions']):
                elite_states.append(state)
                elite_actions.append(action)

        elite_states = torch.FloatTensor(np.array(elite_states))
        elite_actions = torch.FloatTensor(np.array(elite_actions))
        predict_actions = self.forward(elite_states)
        loss = self.loss(predict_actions, elite_actions)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.eps -= self.eps_discount
        if self.eps < 0:
            self.eps = 0

class MainFunc():
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.elite_trajectories_list = []

    def get_trajectories(self):
        trajectories = []
        for _ in range(self.n_trajectories):
            trajectoty = {'states': [], 'actions': [], 'total_reward': 0, 'is_done': False}
            state = self.env.reset()
            for step_trajectory in range(self.max_len):
                trajectoty['states'].append(state)
                action = self.agent.get_action(state)
                trajectoty['actions'].append(action)
                next_state, reward, done, _ = self.env.step(action)
                trajectoty['total_reward'] += reward
                state = next_state

                if done:
                    trajectoty['is_done'] = True
                    break

                if self.visualise:
                    time.sleep(0.02)
                    self.env.render(mode="rgb_array")

                if self.savegif:
                    self.frames.append(self.env.render(mode="rgb_array"))

            trajectories.append(trajectoty)
        if self.savegif:
            self.env.close()
            save_frames_as_gif(self.env, self.frames)
        return trajectories

    def select_elite_trajectories(self, trajectories):
        trajectories_done = []
        for trajectory in trajectories:
            if trajectory['is_done']:
                trajectories_done.append(trajectory)

        result = []
        if len(trajectories_done) == 0:
            if len(self.elite_trajectories_list) > 0:
                for i in self.elite_trajectories_list:
                    result += i
                return result
        else:
            if self.q < (self.q_stop - 0.05):
                self.q += self.q_raise
            else:
                self.q = self.q_stop

        rewards = [trajectory['total_reward'] for trajectory in trajectories_done]
        q_quantile = np.quantile(rewards, self.q)
        elite_trajectories = []
        for trajectory in trajectories_done:
            if trajectory['total_reward'] >= q_quantile:
                elite_trajectories.append(trajectory)

        self.elite_trajectories_list.append(elite_trajectories)
        if len(self.elite_trajectories_list) > self.step_elite:
            del self.elite_trajectories_list[0]


        if len(self.elite_trajectories_list) > 0:
            for i in self.elite_trajectories_list:
                result += i
        return result

    def main_func(self, n_iteration, n_trajectories, q, q_raise, q_stop, step_elite, visualise=False, is_print=False, savegif=False, max_len=998):
        self.n_iteration = n_iteration
        self.n_trajectories = n_trajectories
        self.q = q
        self.q_raise = q_raise
        self.q_stop = q_stop
        self.step_elite = step_elite
        self.visualise = visualise
        self.is_print = is_print
        self.savegif = savegif
        self.frames = []
        self.max_len = max_len
        mean_rewards = []
        for i in range(self.n_iteration):
            trajectories = self.get_trajectories()
            if self.is_print:
                print(f'step: {i}, mean_total_reward: {np.round(np.mean([trajectory["total_reward"] for trajectory in trajectories]), 2)}'+
                      f', mean_done {np.sum([int(trajectory["is_done"]) for trajectory in trajectories]) / n_trajectories}' +
                      f', eps {np.round(self.agent.eps, 2)} q {self.q}')
            mean_rewards.append(np.round(np.mean([trajectory["total_reward"] for trajectory in trajectories]), 2))
            elite_trajectories = self.select_elite_trajectories(trajectories)
            if len(elite_trajectories) > 0:
                self.agent.fit(elite_trajectories)

        return mean_rewards


#action_left_range = -1.
#action_right_range = 1.

#agent_random = RandomAgent(action_left_range, action_right_range)#
#agent_cemdl = CEMDL(action_left_range, action_right_range, 1., eps_discount=1., lr=0.01)

#env = gym.make('MountainCarContinuous-v0')
#mainFunc = MainFunc(env, agent_cemdl)

# train reward 97
#list_rewards = mainFunc.main_func(n_iteration=100, n_trajectories=100, q=0.0, q_raise=0.1, q_stop=0.7, step_elite=1, visualise=False, is_print=False)
#pd.DataFrame(list_rewards, columns=['reward']).to_csv('rewards.csv')

# test
#mainFunc.main_func(1, 1, 0.5, 0, 1,True, False, False)

# savegif
#mainFunc.main_func(1, 1, 0.5, 0, 1,False, False, True)
