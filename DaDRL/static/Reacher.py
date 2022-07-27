import numpy as np

class StaticFns:

    @staticmethod
    def termination_res_fn(env, obs, action, next_obs):
        vec = obs[-3:]
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(action).sum()
        reward = reward_dist + reward_ctrl
        done = False
        return done, reward

    @staticmethod
    def clip_state(env, state):
        high = env.observation_space.high
        low = env.observation_space.low

        b_state = np.clip(state, low, high)

        return b_state



