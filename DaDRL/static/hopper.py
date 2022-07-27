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

        not_done =  np.isfinite(next_obs).all(axis=-1) \
                    * np.abs(next_obs[1:] < 100).all(axis=-1) \
                    * (height > .7) \
                    * (np.abs(ang) < .2)

        done = ~not_done

        return done, reward

    @staticmethod
    def clip_state(env, state):
        high = env.observation_space.high
        low = env.observation_space.low

        b_state = np.concatenate(([state[0]], np.clip(state[1:], low, high)))

        return b_state

