import numpy as np
import math

class StaticFns:
    steps_max = 200
    cur_step = 1

    @staticmethod
    def termination_res_fn(env, obs, act, next_obs):
        tau = 0.02  # seconds between state updates
        x, x_dot, theta, theta_dot = next_obs
        theta_threshold_radians = 12 * 2 * math.pi / 360
        x_threshold = 2.4
        x = x + tau * x_dot
        done = x < -x_threshold \
               or x > x_threshold \
               or theta < -theta_threshold_radians \
               or theta > theta_threshold_radians
        done = bool(done)

        if StaticFns.cur_step < StaticFns.steps_max:
            StaticFns.cur_step += 1
        else:
            StaticFns.cur_step = 1
            done = True

        if not done:
            reward = 1.0
        else:
            reward = 1.0

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