import numpy as np

class StaticFns:
    @staticmethod
    def termination_res_fn(env, obs, act, next_obs):
        posbefore = obs[0]
        posafter, height, ang = next_obs[0:3]
        dt = env.model.opt.timestep * env.frame_skip
        alive_bonus = 1.0
        reward = (posafter - posbefore) / dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(act).sum()

        done = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)

        return done, reward

    @staticmethod
    def clip_state(env, state):
        high = env.observation_space.high
        low = env.observation_space.low

        b_state = np.concatenate(([state[0]], np.clip(state[1:], low, high)))

        return b_state
