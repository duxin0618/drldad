import numpy as np

class StaticFns:

    @staticmethod
    def termination_fn(env, obs, act, next_obs):
        posbefore = obs[8]
        height = obs[0]
        angle = next_obs[1]
        not_done =  np.isfinite(next_obs).all(axis=-1) \
                    * np.abs(next_obs[1:] < 100).all(axis=-1) \
                    * (height > .7) \
                    * (np.abs(angle) < .2)

        done = ~not_done

        return done
