
import numpy as np


class StaticFns:
    steps_max = 1000
    cur_step = 1

    @staticmethod
    def termination_res_fn(env, obs, act, next_obs):

        notdone = np.isfinite(next_obs).all(axis=-1) and (np.abs(next_obs[1]) <= .2)
        done = ~notdone

        if StaticFns.cur_step < StaticFns.steps_max:
            StaticFns.cur_step += 1
        else:
            StaticFns.cur_step = 1
            done = True
        reward = 1
        return done, reward

    @staticmethod
    def resetModel():
        StaticFns.cur_step = 1

    @staticmethod
    def clip_state(env, state):
        high = env.observation_space.high
        low = env.observation_space.low

        b_state = np.clip(state, low, high)

        return b_state