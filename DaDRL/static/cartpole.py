import numpy as np
import math

class StaticFns:
    @staticmethod
    def termination_res_fn(obs, action):
        tau = 0.02  # seconds between state updates
        x, x_dot, theta, theta_dot = obs
        theta_threshold_radians = 12 * 2 * math.pi / 360
        x_threshold = 2.4
        x = x + tau * x_dot
        done = x < -x_threshold \
               or x > x_threshold \
               or theta < -theta_threshold_radians \
               or theta > theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        else:
            reward = 0.0

        return done, reward

    @staticmethod
    def clip_state(env, state):
        high = env.observation_space.high
        low = env.observation_space.low

        b_state = np.clip(state, low, high)

        return b_state