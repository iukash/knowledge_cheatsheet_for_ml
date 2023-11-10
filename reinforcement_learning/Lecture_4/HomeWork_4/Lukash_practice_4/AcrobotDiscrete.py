import numpy as np
from gym import spaces
import gym


class AcrobotDiscrete():
    def __init__(self):
        self.env = gym.make("Acrobot-v1")
        self.eps_zero = 0.001
        self.cosine_sine_min = -1.0
        self.cosine_sine_max = 1.0
        self.angular_velocity_theta1_min = -12.567
        self.angular_velocity_theta1_max = -12.567
        self.angular_velocity_theta2_min = -28.274
        self.angular_velocity_theta2_max = 28.274
        self.action_space = spaces.Discrete(self.env.action_space.n)
        self.len_discrete_state = 21
        self.observation_space = spaces.Discrete(self.len_discrete_state ** 6)

    def get_discrete_value(self, x, min_value, max_value):
        discrete_value = 22

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
        elif 0.5 * min_value <= x < 0.4 * min_value:
            discrete_value = 5
        elif 0.4 * min_value <= x < 0.3 * min_value:
            discrete_value = 6
        elif 0.3 * min_value <= x < 0.2 * min_value:
            discrete_value = 7
        elif 0.2 * min_value <= x < 0.1 * min_value:
            discrete_value = 8
        elif 0.1 * min_value <= x < -1 * self.eps_zero:
            discrete_value = 9
        elif -1 * self.eps_zero <= x <= 1 * self.eps_zero:
            discrete_value = 10
        elif 1 * self.eps_zero < x <= 0.1 * max_value:
            discrete_value = 11
        elif 0.1 * max_value < x <= 0.2 * max_value:
            discrete_value = 12
        elif 0.2 * max_value < x <= 0.3 * max_value:
            discrete_value = 13
        elif 0.3 * max_value < x <= 0.4 * max_value:
            discrete_value = 14
        elif 0.4 * max_value < x <= 0.5 * max_value:
            discrete_value = 15
        elif 0.5 * max_value < x <= 0.6 * max_value:
            discrete_value = 16
        elif 0.6 * max_value < x <= 0.7 * max_value:
            discrete_value = 17
        elif 0.7 * max_value < x <= 0.8 * max_value:
            discrete_value = 18
        elif 0.8 * max_value < x <= 0.9 * max_value:
            discrete_value = 19
        elif x > 0.9 * max_value:
            discrete_value = 20

        return discrete_value

    def discrete_states(self, state):
        discrete_state = np.zeros(len(state))
        discrete_state[0] = self.get_discrete_value(state[0], self.cosine_sine_min, self.cosine_sine_max)
        discrete_state[1] = self.get_discrete_value(state[1], self.cosine_sine_min, self.cosine_sine_max)
        discrete_state[2] = self.get_discrete_value(state[2], self.cosine_sine_min, self.cosine_sine_max)
        discrete_state[3] = self.get_discrete_value(state[3], self.cosine_sine_min, self.cosine_sine_max)
        discrete_state[4] = self.get_discrete_value(state[4], self.angular_velocity_theta1_min, self.angular_velocity_theta1_max)
        discrete_state[5] = self.get_discrete_value(state[5], self.angular_velocity_theta2_min, self.angular_velocity_theta2_max)

        discrete_state_one_array = int(discrete_state[0] + discrete_state[1] * self.len_discrete_state +
                                       discrete_state[2] * self.len_discrete_state ** 2 +
                                       discrete_state[3] * self.len_discrete_state ** 3 +
                                       discrete_state[4] * self.len_discrete_state ** 4 +
                                       discrete_state[5] * self.len_discrete_state ** 5)
        return discrete_state_one_array

    def step(self, action):
        state, reward, done, inf = self.env.step(action)
        return self.discrete_states(state), reward, done, inf

    def reset(self):
        return self.discrete_states(self.env.reset())
