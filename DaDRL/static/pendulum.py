import numpy as np


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

class StaticFns:
    @staticmethod
    def termination_res_fn(obs, action):
        _, th, thdot = obs
        max_torque = 2.
        u = np.clip(action, -max_torque, max_torque)[0]
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        reward = -costs
        return False, reward

    @staticmethod
    def clip_state(env, state):
        high = env.observation_space.high
        low = env.observation_space.low

        b_state = np.clip(state, low, high)

        return b_state