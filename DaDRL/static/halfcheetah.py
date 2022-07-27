import numpy as np

class StaticFns:

    @staticmethod
    def termination_res_fn(env, obs, action, next_obs):

        dt = env.model.opt.timestep * env.frame_skip

        xposbefore = obs[0]
        xposafter = next_obs[0]

        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / dt
        reward = reward_ctrl + reward_run

        done = False
        return done, reward

    @staticmethod
    def clip_state(env, state):
        high = env.observation_space.high
        low = env.observation_space.low

        b_state = np.concatenate(([state[0]], np.clip(state[1:], low, high)))

        return b_state



