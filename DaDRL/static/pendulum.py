import numpy as np


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

class StaticFns:
    steps_max = 200
    cur_step = 1

    @staticmethod
    def termination_res_fn(env, obs, action, next_obs):
        cos, sin, thdot = obs
        th = np.arctan2(sin, cos)
        max_torque = 2.
        u = np.clip(action, -max_torque, max_torque)[0]
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        reward = -costs

        if StaticFns.cur_step < StaticFns.steps_max:
            StaticFns.cur_step += 1
            done = False
        else:
            StaticFns.cur_step = 1
            done = True

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