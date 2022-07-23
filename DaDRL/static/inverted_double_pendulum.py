import sys
import numpy as np
import pdb

class StaticFns:

    @staticmethod
    def termination_fn(obs, act):

        sin1, cos1 = obs[1], obs[3]
        sin2, cos2 = obs[2], obs[4]
        theta_1 = np.arctan2(sin1, cos1)
        theta_2 = np.arctan2(sin2, cos2)

        x = 0.6 * (sin1 + np.sin(theta_1 + theta_2)) + obs[0]
        y = 0.6 * (cos1 + np.cos(theta_1 + theta_2))

        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = obs[6:8]
        vel_penalty = 1e-3 * v1 ** 2 + 5e-3 * v2 ** 2
        alive_bonus = 10

        reward = alive_bonus - dist_penalty - vel_penalty

        done = bool(y <= 1)

        return done, reward

    @staticmethod
    def clip_state(env, state):
        high = env.observation_space.high
        low = env.observation_space.low

        b_state = np.clip(state, low, high)

        return b_state