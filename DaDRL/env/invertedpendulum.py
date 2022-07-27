import gym
import numpy as np

class InvertedPendulumEnv(gym.Wrapper):
    def __init__(self, gym_env_id='InvertedPendulum-v2', target_return=950.0):
        # if gym.__version__ < '0.18.0':
        #     gym_env_id = 'InvertedPendulum-v2'
        # elif gym.__version__ >= '0.20.0':
        #     gym_env_id = 'InvertedPendulum-v2'
        gym.logger.set_level(40)  # Block warning
        super(InvertedPendulumEnv, self).__init__(env=gym.make(gym_env_id))

        self.env_num = 1  # the env number of VectorEnv is greater than 1
        self.env_name = gym_env_id  # the name of this env.
        self.max_step = 1000  # the max step of each episode
        self.state_dim = 4  # feature number of state
        self.action_dim = 1  # feature number of action
        self.if_discrete = False  # discrete action or continuous action
        self.target_return = target_return  # episode return is between (-1600, 0)

        print(f'\n| {self.__class__.__name__}: InvertedPendulum-v2 Env set its action space as (-3, +3).'
              f'\n| And we scale the action, and set the action space as (-1, +1).'
              f'\n| So do not use your policy network on raw env directly.')

    def reset(self):
        return self.env.reset().astype(np.float32)

    def step(self, action: np.ndarray):
        # PendulumEnv set its action space as (-3, +3). It is bad.  # https://github.com/openai/gym/wiki/Pendulum-v0
        # I suggest to set action space as (-1, +1) when you design your own env.
        state, reward, done, info_dict = self.env.step(action * 3)  # state, reward, done, info_dict
        return state.astype(np.float32), reward, done, info_dict
