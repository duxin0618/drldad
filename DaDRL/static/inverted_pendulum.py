import sys
import numpy as np
import pdb

class StaticFns:

    @staticmethod
    def termination_fn(obs, act):

        notdone = np.isfinite(obs).all(axis=-1) and (np.abs(obs[1]) <= .2)
        done = ~notdone
        reward = 1
        return done, reward

    @staticmethod
    def clip_state(env, state):
        high = env.observation_space.high
        low = env.observation_space.low

        b_state = np.clip(state, low, high)

        return b_state