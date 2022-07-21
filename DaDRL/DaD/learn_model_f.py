from sklearnex import patch_sklearn
patch_sklearn() #启动加速补丁

import os, sys
import numpy as np
from DaDRL.DaD.DaggerTrajactory_f import DaDModel
from DaDRL.DaD.ModelNet import DynamicsModelDeltaWrapper

import torch

GAIL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(GAIL_PATH)

from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge


class Dagger:

    def __init__(self, args):


        # 线性学习网络
        # learner = MLPRegressor(
        #     hidden_layer_sizes=hidden_layer_size, activation='tanh', solver='adam', alpha=0.0001, batch_size='auto',
        #     learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=50000, shuffle=True,
        #     random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        #     early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        # learner = MLPRegressor(hidden_layer_sizes=(10, 2),
        #              activation='tanh', alpha=1e-3, max_iter=int(1e4), warm_start=False)

        learner = Ridge(alpha=1e-4, fit_intercept=True)
        # learner = KernelRidge(alpha=1e-4)

        self.ModelNet = DynamicsModelDeltaWrapper(learner)
        self.iters = None
        self.dad = DaDModel(self.ModelNet)

        self.maxsteps = args.max_step
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.if_discrete = args.if_discrete

        self.gamma = getattr(args, "gamma", 0.99)
        self.env_num = getattr(args, "env_num", 1)
        self.reward_scale = getattr(args, "reward_scale", 1.0)
        self.if_use_old_traj = getattr(args, "if_use_old_traj", False)
        self.if_off_policy = False
        self.cwd = args.cwd
        self.traj_list = [
            [list() for _ in range(4 if self.if_off_policy else 5)]
            for _ in range(self.env_num)
        ]
        gpu_id = args.learner_gpus
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.it_steps = 1

    def doLearn(self, states, actions, states_next):

        state_train = np.stack(states, axis=2)
        # action_train = np.dstack(self.actions).transpose((1, 0, 2))
        action_train = np.stack(actions, axis=2)
        state_next_train = np.stack(states_next, axis=2)
        self.optimize_learner_dad(state_train, action_train, state_next_train)

    # train model
    def train(self, buffer):
        buf_state, buf_reward, buf_mask, buf_action, buf_noise = [ten for ten in buffer]
        if self.if_discrete:
            self.action_dim = 1
        len = buf_state.shape[0]
        train_sample = (len - len % self.maxsteps)
        self.it_traj = train_sample / self.maxsteps

        """merge state and rewards"""

        # states_rewards = np.concatenate((buf_reward, buf_state), axis=1)
        """"""
        states = buf_state[:train_sample, :].reshape(int(self.it_traj), int(self.maxsteps), self.state_dim)[:, :-1]
        actions = buf_action[:train_sample, :].reshape(int(self.it_traj), int(self.maxsteps), self.action_dim)[:, :-1]
        states_next = buf_state[:train_sample, :].reshape(int(self.it_traj), int(self.maxsteps), self.state_dim)[:, 1:]
        self.doLearn(states, actions, states_next)
        self.dagger_trajectory()
        return self.dad.min_train_error, self.dad.initial_model_error

    # explore_env
    def explore_env(self, env, target_step, buffer, raw_reward, act, fc):

        predict = self.dad.min_train_error_model.predict
        buf_state, buf_reward, buf_mask, buf_action, buf_noise = [ten for ten in buffer]
        # sample random number init states
        init_states_index = np.random.choice(np.array(buf_state).shape[0], 4)
        init_states = buf_state[init_states_index]
        traj_list = list()
        last_done = [
            0,
        ]

        step_i = 0

        done = False
        for epoch in range(4):
            state = init_states[epoch]
            idx = init_states_index[epoch]
            while not done:
                if not torch.is_tensor(state):
                    ten_s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                else:
                    ten_s = state
                ten_a, ten_n = [
                    ten.cpu() for ten in act(ten_s.to(self.device))
                ]  # different
                action = ten_a.tanh()[0].numpy()
                _ob_s = np.expand_dims(state, axis=0)
                _ac_s = np.expand_dims(action, axis=0)

                _ob_next = predict(_ob_s, _ac_s)[0]

                reward = buf_reward[idx]
                reward = torch.as_tensor(reward, dtype=torch.float32).unsqueeze(0)
                done = torch.as_tensor(fc.termination_fn(state, action, _ob_next))

                traj_list.append((ten_s, reward, done, ten_a, ten_n))  # different
                idx += 1
                step_i += 1

        last_done[0] = step_i
        return self.convert_trajectory(traj_list, last_done), step_i  # traj_list

    def convert_trajectory(self, buf_items, last_done):  # [ElegantRL.2022.01.01]
        """convert trajectory (env exploration type) to trajectory (replay buffer type)

        convert `other = concat((      reward, done, ...))`
        to      `other = concat((scale_reward, mask, ...))`

        :param traj_list: `traj_list = [(tensor_state, other_state), ...]`
        :return: `traj_list = [(tensor_state, other_state), ...]`
        """
        # assert len(buf_items) == step_i
        # assert len(buf_items[0]) in {4, 5}
        # assert len(buf_items[0][0]) == self.env_num

        buf_items = list(
            map(list, zip(*buf_items))
        )  # state, reward, done, action, noise
        # assert len(buf_items) == {4, 5}
        # assert len(buf_items[0]) == step
        # assert len(buf_items[0][0]) == self.env_num

        """stack items"""
        buf_items[0] = torch.stack(buf_items[0])
        buf_items[3:] = [torch.stack(item) for item in buf_items[3:]]

        if len(buf_items[3].shape) == 2:
            buf_items[3] = buf_items[3].unsqueeze(2)

        if self.env_num > 1:
            buf_items[1] = (torch.stack(buf_items[1]) * self.reward_scale).unsqueeze(2)
            buf_items[2] = ((~torch.stack(buf_items[2])) * self.gamma).unsqueeze(2)
        else:
            buf_items[1] = (
                (torch.tensor(buf_items[1], dtype=torch.float32) * self.reward_scale)
                .unsqueeze(1)
                .unsqueeze(2)
            )
            buf_items[2] = (
                ((1 - torch.tensor(buf_items[2], dtype=torch.float32)) * self.gamma)
                .unsqueeze(1)
                .unsqueeze(2)
            )
        # assert all([buf_item.shape[:2] == (step, self.env_num) for buf_item in buf_items])

        """splice items"""
        for j in range(len(buf_items)):
            cur_item = list()
            buf_item = buf_items[j]

            for env_i in range(self.env_num):
                last_step = last_done[env_i]

                pre_item = self.traj_list[env_i][j]
                if len(pre_item):
                    cur_item.append(pre_item)

                cur_item.append(buf_item[:last_step, env_i])

                if self.if_use_old_traj:
                    self.traj_list[env_i][j] = buf_item[last_step:, env_i]

            buf_items[j] = torch.vstack(cur_item)

        # on-policy:  buf_item = [states, rewards, dones, actions, noises]
        # off-policy: buf_item = [states, rewards, dones, actions]
        # buf_items = [buf_item, ...]
        return buf_items

    def dagger_trajectory(self):
        self.dad.dagger_trajectory()


    def doDagger(self, states, actions, states_next, iters):
        state_train = np.stack(states, axis=2)
        action_train = np.stack(actions, axis=2)
        state_next_train = np.stack(states_next, axis=2)
        self.dagger_data_dad(state_train, action_train, state_next_train, iters)



    # 利用交叉验证的方法，划分成训练集和测试集，训练集进行训练DaD模型，测试集进行测试model的准确度
    def optimize_learner_dad(self, states, actions, state_next):

        # learn
        self.dad.learn(states, actions, state_next)

    def dagger_data_dad(self, states, actions, state_next, iters):
        self.dad.dagger_data(states, actions, state_next, iters)


    def get_data(self):
        return self.dad








