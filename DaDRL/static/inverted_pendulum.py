
import numpy as np


class StaticFns:

    @staticmethod
    def termination_res_fn(env, obs, act, next_obs):

        notdone = np.isfinite(next_obs).all(axis=-1) and (np.abs(next_obs[1]) <= .2)
        done = ~notdone
        reward = 1
        return done, reward

    @staticmethod
    def clip_state(env, state):
        high = env.observation_space.high
        low = env.observation_space.low

        b_state = np.clip(state, low, high)

        return b_state