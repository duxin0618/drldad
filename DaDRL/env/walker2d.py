import numpy as np
import gym



class Walker2dEnv(gym.Wrapper):  # [ElegantRL.2022.04.04]
    def __init__(self, gym_env_id='Walker2d-v2', target_return=4000):

        gym_env_id = gym_env_id
        gym.logger.set_level(40)  # Block warning
        super(Walker2dEnv, self).__init__(env=gym.make(gym_env_id))

        # from elegantrl.envs.Gym import get_gym_env_info
        # get_gym_env_info(env, if_print=True)  # use this function to print the env information
        self.env_num = 1  # the env number of VectorEnv is greater than 1
        self.env_name = gym_env_id  # the name of this env.
        self.max_step = 1000  # the max step of each episode
        self.state_dim = 17  # feature number of state
        self.action_dim = 6  # feature number of action
        self.if_discrete = False  # discrete action or continuous action
        self.target_return = target_return  # episode return is between (-1600, 0)

        print(f'\n| {self.__class__.__name__}: Walker2d-v2 Env set its state_dim is 17->18 as train dad model.')

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return self.state_vector()

    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        return ob

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self.state_vector()
