import numpy as np
from gym import spaces
import gym


class CartPole_v_hw4():
    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.eps_zero = 0.001
        self.cart_position_min = -2.4
        self.cart_position_max = 2.4
        self.cart_velocity_min = -1.7
        self.cart_velocity_max = 2.0
        self.pole_angle_min = -0.21
        self.pole_angle_max = 0.21
        self.pole_angular_velocity_min = -3.1
        self.pole_angular_velocity_max = 2.8
        self.action_space = spaces.Discrete(self.env.action_space.n)
        self.len_discrete_state = 31
        self.observation_space = spaces.Discrete(self.len_discrete_state ** 4)

    def get_discrete_value(self, x, min_value, max_value):
        discrete_value = 11

        if x < 0.9 * min_value:
            discrete_value = 0
        elif 0.9 * min_value <= x < 0.8 * min_value:
            discrete_value = 1
        elif 0.8 * min_value <= x < 0.7 * min_value:
            discrete_value = 2
        elif 0.7 * min_value <= x < 0.6 * min_value:
            discrete_value = 3
        elif 0.6 * min_value <= x < 0.5 * min_value:
            discrete_value = 4
        elif 0.5 * min_value <= x < 0.45 * min_value:
            discrete_value = 5
        elif 0.45 * min_value <= x < 0.4 * min_value:
            discrete_value = 6
        elif 0.4 * min_value <= x < 0.35 * min_value:
            discrete_value = 7
        elif 0.35 * min_value <= x < 0.3 * min_value:
            discrete_value = 8
        elif 0.3 * min_value <= x < 0.25 * min_value:
            discrete_value = 9
        elif 0.25 * min_value <= x < 0.2 * min_value:
            discrete_value = 10
        elif 0.2 * min_value <= x < 0.15 * min_value:
            discrete_value = 11
        elif 0.15 * min_value <= x < 0.1 * min_value:
            discrete_value = 12
        elif 0.1 * min_value <= x < 0.05 * min_value:
            discrete_value = 13
        elif 0.05 * min_value <= x < -1 * self.eps_zero:
            discrete_value = 14
        elif -1 * self.eps_zero <= x <= 1 * self.eps_zero:
            discrete_value = 15
        elif 1 * self.eps_zero < x <= 0.05 * max_value:
            discrete_value = 16
        elif 0.05 * max_value < x <= 0.1 * max_value:
            discrete_value = 17
        elif 0.1 * max_value < x <= 0.15 * max_value:
            discrete_value = 18
        elif 0.15 * max_value < x <= 0.2 * max_value:
            discrete_value = 19
        elif 0.2 * max_value < x <= 0.25 * max_value:
            discrete_value = 20
        elif 0.25 * max_value < x <= 0.3 * max_value:
            discrete_value = 21
        elif 0.3 * max_value < x <= 0.35 * max_value:
            discrete_value = 22
        elif 0.35 * max_value < x <= 0.4 * max_value:
            discrete_value = 23
        elif 0.4 * max_value < x <= 0.45 * max_value:
            discrete_value = 24
        elif 0.45 * max_value < x <= 0.5 * max_value:
            discrete_value = 25
        elif 0.5 * max_value < x <= 0.6 * max_value:
            discrete_value = 26
        elif 0.6 * max_value < x <= 0.7 * max_value:
            discrete_value = 27
        elif 0.7 * max_value < x <= 0.8 * max_value:
            discrete_value = 28
        elif 0.8 * max_value < x <= 0.9 * max_value:
            discrete_value = 29
        elif x > 0.9 * max_value:
            discrete_value = 30

        return discrete_value

    def discrete_states(self, state):
        discrete_state = np.zeros(len(state))
        discrete_state[0] = self.get_discrete_value(state[0], self.cart_position_min, self.cart_position_max)
        discrete_state[1] = self.get_discrete_value(state[1], self.cart_velocity_min, self.cart_velocity_max)
        discrete_state[2] = self.get_discrete_value(state[2], self.pole_angle_min, self.pole_angle_max)
        discrete_state[3] = self.get_discrete_value(state[3], self.pole_angular_velocity_min,
                                                    self.pole_angular_velocity_max)

        discrete_state_one_array = int(discrete_state[0] + discrete_state[1] * self.len_discrete_state +
                                       discrete_state[2] * self.len_discrete_state ** 2 + discrete_state[
                                           3] * self.len_discrete_state ** 3)
        # print([discrete_state[0], discrete_state[1], discrete_state[2], discrete_state[3]])
        return discrete_state_one_array

    def step(self, action):
        state, reward, done, inf = self.env.step(action)
        return self.discrete_states(state), reward, done, inf

    def reset(self):
        return self.discrete_states(self.env.reset())