import sys
import numpy as np
import pdb

class StaticFns:

    @staticmethod
    def termination_res_fn(env, obs, act, next_obs):

        sin1, cos1 = next_obs[1], next_obs[3]
        sin2, cos2 = next_obs[2], next_obs[4]
        theta_1 = np.arctan2(sin1, cos1)
        theta_2 = np.arctan2(sin2, cos2)

        x = 0.6 * (sin1 + np.sin(theta_1 + theta_2)) + next_obs[0]
        y = 0.6 * (cos1 + np.cos(theta_1 + theta_2))

        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = next_obs[6:8]
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