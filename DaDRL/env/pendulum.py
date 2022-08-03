
import gym
import numpy as np

class PendulumEnv(gym.Wrapper):  # [ElegantRL.2022.04.04]
    def __init__(self, gym_env_id='Pendulum-v1', target_return=-200):
        if gym.__version__ < '0.18.0':
            gym_env_id = 'Pendulum-v0'
        elif gym.__version__ >= '0.20.0':
            gym_env_id = 'Pendulum-v1'
        gym.logger.set_level(40)  # Block warning
        super(PendulumEnv, self).__init__(env=gym.make(gym_env_id))

        # from elegantrl.envs.Gym import get_gym_env_info
        # get_gym_env_info(env, if_print=True)  # use this function to print the env information
        self.env_num = 1  # the env number of VectorEnv is greater than 1
        self.env_name = gym_env_id  # the name of this env.
        self.max_step = 200  # the max step of each episode
        self.state_dim = 3  # feature number of state
        self.action_dim = 1  # feature number of action
        self.if_discrete = False  # discrete action or continuous action
        self.target_return = target_return  # episode return is between (-1600, 0)

        print(f'\n| {self.__class__.__name__}: Pendulum Env set its action space as (-2, +2).'
              f'\n| And we scale the action, and set the action space as (-1, +1).'
              f'\n| So do not use your policy network on raw env directly.')

    def reset(self):
        return self.env.reset().astype(np.float32)

    def step(self, action: np.ndarray):
        # PendulumEnv set its action space as (-2, +2). It is bad.  # https://github.com/openai/gym/wiki/Pendulum-v0
        # I suggest to set action space as (-1, +1) when you design your own env.
        state, reward, done, info_dict = self.env.step(action * 2)  # state, reward, done, info_dict
        return state.astype(np.float32), reward, done, info_dict

