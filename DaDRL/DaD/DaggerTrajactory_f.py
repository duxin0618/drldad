from copy import deepcopy
import numpy as np
from DaDRL.DaD import pyhelpers
import math
import time

__author__ = 'DX'


class DaDModel(object):

    def __init__(self, learner, traj_pwr_filter = 1, rollout_err_filter = 1, cwd=None):
        self.TRAJ_PWR_SCALE = traj_pwr_filter  # 静态阈值 ： 缩小RMS
        self.TRAIN_ROLLOUT_ERR_SCALE = rollout_err_filter  # 动态阈值：缩小RMS
        # store reference to learner
        self.min_train_error = math.inf
        self.backward_mean_err = math.inf
        self.learner = learner
        self.min_train_error_model = learner

        # record iter error
        self.min_iter_error_gather = []
        self.min_iter_recession_error_gather = []
        self.min_iter_all_model_error = math.inf


    def learn(self, state_s, action_s, state_s_next):
        # 数据变形

        # state_t 由[timesteps , dim_state , num_traj] ==> [num_traj*timesteps, dim_data]
        self.state_t = pyhelpers.tensor_to_dataset(state_s)
        self.state_t1 = pyhelpers.tensor_to_dataset(state_s_next)
        self.action_t = pyhelpers.tensor_to_dataset(action_s)

        # 原数据集
        self.state_s = state_s
        self.action_s = action_s
        self.state_s_next = state_s_next

        # 计算轨迹的均方误差， 初始是与0进行运算
        self.mean_traj_pwr = np.mean(pyhelpers.rms_error(state_s, np.zeros(state_s.shape)))

        # 拟合(s_t, a_t, s_{t+1})  这里的学习器使用的是sklearn的Ridge

        self.learner.fit(self.state_t, self.action_t, self.state_t1)
        self.initial_model = deepcopy(self.learner)

        # 初始化训练误差
        _, errors = self.test(state_s, action_s, state_s_next, self.learner)  # 返回的是预测状态 与 RMS误差
        self.mean_train_rollout_err = np.mean(errors)
        self.initial_model_error = self.mean_train_rollout_err
        self.min_train_error = self.mean_train_rollout_err
        self.min_train_error_model = deepcopy(self.learner)

        print(' Init error: {0:.4g}'.format(self.mean_train_rollout_err))


        # print("min_train_error", self.min_train_error)

    def dagger_trajectory(self, dagger_iters=20):
        t1 = time.time()
        self.datasize = self.state_s.shape[0] * self.state_s.shape[2]
        self.MAX_TRAIN_SIZE = self.datasize * 10
        self.MAX_DATA_SIZE = math.ceil(self.datasize * 20)
        # 控制最多5倍
        self.MAX_DATA_LEN = self.MAX_DATA_SIZE + self.datasize
        self.new_sample_num = 0
        # 实时数据集合大小
        self.dataaddsize = self.datasize

        self.new_state_t = np.empty((1, self.state_t.shape[1]))
        self.new_state_t1 = np.empty((1, self.state_t.shape[1]))
        self.new_action_t = np.empty((1, self.action_t.shape[1]))

        state_t1_gt = self.state_t1.copy()  # these will be ground truth targets
        state_0s = self.state_s[0, :, :]  # [dim_state, num_traj] 所有初始的states
        state_next_0s = self.state_s_next[0, :, :]


        # Start the DaD Main loop
        train_errors = []

        self.new_sample_indexs = []
        # 轨迹数没有变化，时间步骤变长了，不断地最初的(s,a)去获取误差较小的(s,a)加入到(S,A)中
        # print("原始样本个数：", self.MAX_TRAIN_SIZE)

        # record all error with iter
        cur_iters_error_gather = []
        cur_iters_error_gather.append(self.min_train_error)

        # record recession error with iter
        cur_iters_error_recession_gather = []
        cur_iters_error_recession_gather.append(self.min_train_error)

        for i in range(1, dagger_iters + 1):  # iteration
            # print(">DaD Iteration: {}/{}".format(i, dagger_iters))
            # 获取预测的状态信息，传入初始state和动作集
            state_pred = self._rollout_model(state_0s, self.action_s, state_next_0s)
            # 计算RMS误差
            self.mean_train_rollout_err = np.mean(pyhelpers.rms_error(state_pred[:-1, :, :], self.state_s))
            cur_iters_error_gather.append(self.mean_train_rollout_err)
            # 更新误差
            if self.mean_train_rollout_err < self.min_train_error:

                self.min_train_error = self.mean_train_rollout_err
                self.min_train_error_model = deepcopy(self.learner)
                cur_iters_error_recession_gather.append(self.min_train_error)
            # print("min_train_error", self.min_train_error)
            # 记录误差
            train_errors.append(self.mean_train_rollout_err)
            # 打印提示信息
            # print()
            # print(' Training error: {0:.4g}'.format(self.mean_train_rollout_err))
            # print(' Min error: {0:.4g}'.format(self.min_train_error))

            state_t_hat = pyhelpers.tensor_to_dataset(state_pred[:-1, :, :])  # 去除最后一个状态T+1->T
            keep_inds = self._remove_large_err_samples(state_t1_gt, state_t_hat)  # (T,T) 去除误差较大的时间T时刻的样本点
            num_total = self.state_t.shape[0]  # 统计时间步个数 T
            num_kept = keep_inds.size  # 统计较小的预测state与目标state的样本下标
            # 打印提示信息
            self.new_sample_indexs = np.concatenate((self.new_sample_indexs, keep_inds))

            # np.concatenate 合并, 默认axis=0, 增加了时间步的长度
            # if ((num_total + num_kept) > self.MAX_DATA_SIZE):
            #     num_kept = self.MAX_DATA_SIZE - num_total
            self.new_sample_num += num_kept
            # print("keep_inds 新加样本点数：", num_kept)
            # print(" Keeping {:d}/{:d}={:3.2f}% of the data".format(num_kept, num_total,
            #                                                        float(num_kept) / float(num_total) * 100))
            # 收集预测的信息
            if i == 1:
                self.new_state_t = state_t_hat[keep_inds[0: num_kept]]
                self.new_state_t1 = state_t1_gt[keep_inds[0: num_kept]]
                self.new_action_t = self.action_t[keep_inds[0: num_kept]]

            else:
                self.new_state_t = np.concatenate((self.new_state_t, state_t_hat[keep_inds[0: num_kept]]))
                self.new_state_t1 = np.concatenate((self.new_state_t1, state_t1_gt[keep_inds[0: num_kept]]))
                self.new_action_t = np.concatenate((self.new_action_t, self.action_t[keep_inds[0: num_kept]]))

            self.state_t = np.concatenate((self.state_t, state_t_hat[keep_inds[0: num_kept]]))
            self.state_t1 = np.concatenate((self.state_t1, state_t1_gt[keep_inds[0: num_kept]]))
            self.action_t = np.concatenate((self.action_t, self.action_t[keep_inds[0: num_kept]]))

            train_state_t = self.state_t
            train_state_t1 = self.state_t1
            train_action_t = self.action_t

            self.dataaddsize = self.state_t.shape[0]
            if self.dataaddsize > self.MAX_TRAIN_SIZE:
                perm = np.random.choice(range(0, int(self.dataaddsize)), self.MAX_TRAIN_SIZE, replace=False)
                train_state_t = self.state_t[perm]
                train_state_t1 = self.state_t1[perm]
                train_action_t = self.action_t[perm]
            # 再去学习一遍, 优化动态模型
            self.learner.fit(train_state_t, train_action_t,  train_state_t1)
            # print(' Dataset Size: {:d}.'.format(self.state_t.shape[0]))
            # print("self.dataaddsize", self.dataaddsize)
            if self.dataaddsize >= self.MAX_DATA_SIZE:
                perm = np.random.choice(range(0, int(self.dataaddsize)), self.MAX_TRAIN_SIZE, replace=False)
                self.state_t = self.state_t[perm]
                self.state_t1 = self.state_t1[perm]
                self.action_t = self.action_t[perm]
            # print("Datasize :" , self.dataaddsize)


        self.backward_mean_err = np.mean(train_errors)
        if self.min_train_error < self.min_iter_all_model_error:
            self.min_iter_all_model_error = self.min_train_error
            self.min_iter_error_gather = cur_iters_error_gather
            self.min_iter_recession_error_gather = cur_iters_error_recession_gather
        t2 = time.time()
        print("use time: ",t2-t1)


    def dagger_data(self, state_s, action_s, state_s_next, dagger_iters = 20):

        self.datasize = state_s.shape[0] * state_s.shape[2]
        self.MAX_TRAIN_SIZE = self.datasize * 2
        self.MAX_DATA_SIZE = math.ceil(self.datasize * 4)
        # 控制最多5倍
        self.MAX_DATA_LEN = self.MAX_DATA_SIZE + self.datasize
        self.new_sample_num = 0
        # 实时数据集合大小
        self.dataaddsize = self.datasize

        self.state_t_d = pyhelpers.tensor_to_dataset(state_s)
        self.state_t1_d = pyhelpers.tensor_to_dataset(state_s_next)
        self.action_t_d = pyhelpers.tensor_to_dataset(action_s)

        self.new_state_t = np.empty((1, self.state_t_d.shape[1]))
        self.new_state_t1 = np.empty((1, self.state_t1_d.shape[1]))
        self.new_action_t = np.empty((1, self.action_t_d.shape[1]))

        state_t1_gt = self.state_t1_d.copy()  # these will be ground truth targets
        state_0s = state_s[0, :, :]  # [dim_state, num_traj] 所有初始的states
        state_next_0s = state_s_next[0, :, :]
        # Start the DaD Main loop
        train_errors = []
        # 轨迹数没有变化，时间步骤变长了，不断地最初的(s,a)去获取误差较小的(s,a)加入到(S,A)中
        # print("原始样本个数：", self.MAX_TRAIN_SIZE)

        for i in range(1, dagger_iters + 1):  # iteration
            # print(">DaD Iteration: {}/{}".format(i, dagger_iters))

            # 获取预测的状态信息，传入初始state和动作集
            state_pred = self._rollout_model(state_0s, action_s, state_next_0s)
            # 计算RMS误差
            self.mean_train_rollout_err = np.mean(pyhelpers.rms_error(state_pred[:-1, :, :], state_s))
            # 更新误差
            if self.mean_train_rollout_err < self.min_train_error:
                self.min_train_error = self.mean_train_rollout_err
            # print("min_train_error", self.min_train_error)
            # 记录误差
            train_errors.append(self.mean_train_rollout_err)
            # 打印提示信息
            # print(' Training error: {0:.4g}'.format(self.mean_train_rollout_err))

            state_t_hat = pyhelpers.tensor_to_dataset(state_pred[:-1, :, :])  # 去除最后一个状态T+1->T
            keep_inds = self._remove_large_err_samples(state_t1_gt, state_t_hat)  # (T,T) 去除误差较大的时间T时刻的样本点
            num_kept = keep_inds.size  # 统计较小的预测state与目标state的样本下标
            # 打印提示信息


            # np.concatenate 合并, 默认axis=0, 增加了时间步的长度
            # if ((num_total + num_kept) > self.MAX_DATA_SIZE):
            #     num_kept = self.MAX_DATA_SIZE - num_total
            self.new_sample_num += num_kept
            # print("keep_inds 新加样本点数：", num_kept)
            # print(" Keeping {:d}/{:d}={:3.2f}% of the data".format(num_kept, num_total,
            #                                                        float(num_kept) / float(num_total) * 100))

            # 收集预测的信息
            if i == 1:
                self.new_state_t = state_t_hat[keep_inds[0: num_kept]]
                self.new_state_t1 = state_t1_gt[keep_inds[0: num_kept]]
                self.new_action_t = self.action_t_d[keep_inds[0: num_kept]]

            else:
                self.new_state_t = np.concatenate((self.new_state_t, state_t_hat[keep_inds[0: num_kept]]))
                self.new_state_t1 = np.concatenate((self.new_state_t1, state_t1_gt[keep_inds[0: num_kept]]))
                self.new_action_t = np.concatenate((self.new_action_t, self.action_t_d[keep_inds[0: num_kept]]))

            self.state_t_d = np.concatenate((self.state_t_d, state_t_hat[keep_inds[0: num_kept]]))
            self.state_t1_d = np.concatenate((self.state_t1_d, state_t1_gt[keep_inds[0: num_kept]]))
            self.action_t_d = np.concatenate((self.action_t_d, self.action_t_d[keep_inds[0: num_kept]]))

            train_state_t = self.state_t_d
            train_state_t1 = self.state_t1_d
            train_action_t = self.action_t_d

            self.dataaddsize = self.state_t_d.shape[0]

            if self.dataaddsize > self.MAX_TRAIN_SIZE:
                perm = np.random.choice(range(0, int(self.dataaddsize)), self.MAX_TRAIN_SIZE, replace=False)
                train_state_t = self.state_t_d[perm]
                train_state_t1 = self.state_t1_d[perm]
                train_action_t = self.action_t_d[perm]
            # 再去学习一遍, 优化动态模型
            self.learner.fit(train_state_t, train_action_t, train_state_t1)

            if self.dataaddsize >= self.MAX_DATA_SIZE:
                break

        if self.dataaddsize > self.MAX_DATA_LEN:
            perm = np.random.choice(range(0, int(self.dataaddsize)), self.MAX_DATA_LEN, replace=False)
            self.state_t_d = self.state_t_d[perm]
            self.state_t1_d = self.state_t1_d[perm]
            self.action_t_d = self.action_t_d[perm]
            self.dataaddsize = self.MAX_DATA_LEN

        self.backward_mean_err = np.mean(train_errors)

        return np.mean(train_errors)



    def test(self, state_test, action_test, state_s_next, learner=None):
        if learner is None:
            learner = self.initial_model
        state_0s_test = state_test[0, :, :]
        state_next_0s = state_s_next[0, :, :]
        state_pred_test = self._rollout_model(state_0s_test, action_test, state_next_0s, learner)
        test_rollout_err = pyhelpers.rms_error(state_pred_test[:-1,:,:],  state_test)
        return state_pred_test, test_rollout_err

    def _rollout_model(self, state_0s, action_s, state_next_0s, learner=None):
        T = action_s.shape[0]  # 获取timesteps个数
        num_traj = action_s.shape[2]  # 获取轨迹个数
        dim_state = state_0s.shape[0]  #获取状态空间个数
        if learner is None:
            learner = self.learner
        # prevent '.' lookups for speed
        predict = learner.predict  # 获取预测器
        # 为了使得下标统一，预测的状态的时间步为T+1
        predictions = np.zeros((T+1, dim_state, num_traj)) + np.NaN
        # initialize the current state_t_now
        state_t_now = state_0s.T  # make it [num_traj, dim_state]
        predictions[0,:,:] = state_t_now.T  # make it [dim_state, num_traj ]
        i = 0
        for t in range(0, T):  # 遍历每一个时间步
            i += 1
            action_t_now = action_s[t, :, :].T
            state_t1= predict(state_t_now, action_t_now) # should be [num_traj, dim_state] 返回的是下一状态
            predictions[t+1, :, :] = state_t1.T
            state_t_now = state_t1    # 不断循环
        # 返回预测的状态
        return predictions

    def _remove_large_err_samples(self, state_tgt, state_that):

        error = state_tgt - state_that
        l2_err = np.sqrt(np.sum(error * error, 1))
        keep_inds = np.where(
            np.logical_and(l2_err < self.TRAJ_PWR_SCALE*self.mean_traj_pwr, l2_err < self.TRAIN_ROLLOUT_ERR_SCALE *self.min_train_error))[0]
        return keep_inds



