import numpy as np
import gym



class HalfCheetahEnv(gym.Wrapper):  # [ElegantRL.2022.04.04]
    def __init__(self, gym_env_id='HalfCheetah-v2', target_return=4800):

        gym_env_id = 'HalfCheetah-v2'
        gym.logger.set_level(40)  # Block warning
        super(HalfCheetahEnv, self).__init__(env=gym.make(gym_env_id))

        # from elegantrl.envs.Gym import get_gym_env_info
        # get_gym_env_info(env, if_print=True)  # use this function to print the env information
        self.env_num = 1  # the env number of VectorEnv is greater than 1
        self.env_name = gym_env_id  # the name of this env.
        self.max_step = 1000  # the max step of each episode
        self.state_dim = 17  # feature number of state
        self.action_dim = 6  # feature number of action
        self.if_discrete = False  # discrete action or continuous action
        self.target_return = target_return  # episode return is between (-1600, 0)

        self.cur_step_number = 0

        print(f'\n| {self.__class__.__name__}: HalfCheetah-v2 Env set its state_dim is 17 -> 18 as train dad model.')

    def step(self, action):

        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        if self.cur_step_number < self.max_step:
            done = False
        else:
            self.cur_step_number = -1
            done = True

        self.cur_step_number += 1
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return self.state_vector()

    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        return ob

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self.state_vector()
