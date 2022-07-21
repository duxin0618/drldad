#
# Learner Wrapper helper classes for Data as Demonstrator (DaD) repository.
#
from sklearnex import patch_sklearn
patch_sklearn() #启动加速补丁

import numpy as np
import DaDRL.DaD.pyhelpers as pyh
from sklearn.kernel_approximation import RBFSampler

class DynamicsModelWrapper(object):


    def __init__(self, learner):
        self.learner = learner

    def fit(self, state_t, action_t, state_t1):

        state_t = pyh.ensure_2d(state_t)
        action_t = pyh.ensure_2d(action_t)
        inputs = np.hstack((state_t, action_t))
        self.learner.fit(inputs, state_t1)

    def predict(self, state_t, action_t):
        if len(state_t.shape) == 1:
            inputs = np.expand_dims(np.hstack((state_t, action_t)), axis=0)
            state_t1 = self.learner.predict(inputs)
            return state_t1.ravel()
        inputs = np.hstack((state_t, action_t))
        state_t1 = self.learner.predict(inputs)

        return state_t1


class DynamicsModelDeltaWrapper(object):


    def __init__(self, learner):
        self.learner = learner
        self.rbf_feature = RBFSampler(gamma=1, random_state=1, n_components=100)

    def fit(self, state_t, action_t, state_t1):
        state_t = pyh.ensure_2d(state_t)  # 确保是两维度的
        state_t1 = pyh.ensure_2d(state_t1)
        action_t = pyh.ensure_2d(action_t)
        # 输入时间T时刻的state与action，预测state差异
        inputs = np.hstack((state_t, action_t))
        # consider reward
        if state_t1.shape != state_t.shape:
            state_t = np.pad(state_t, pad_width=((0, 0), (1, 0)), mode='constant', constant_values=0)
        d_state = np.subtract(state_t1, state_t)
        # divide state and reward

        X = self.rbf_feature.fit_transform(inputs)

        self.learner.fit(X, d_state)


    # 返回下一状态
    def predict(self, state_t, action_t):
        # if len(state_t.shape) == 1:
        #     print("#####################################hhhhhhhh")
        #     inputs = np.expand_dims(np.hstack((state_t, action_t)), axis=0)
        #     d_state = self.learner.predict(inputs)
        #     state_t1 = state_t + d_state
        #     return state_t1.ravel()

        inputs = np.hstack((state_t, action_t))
        X = self.rbf_feature.fit_transform(inputs)

        d_state = self.learner.predict(X)
        if np.shape(state_t) != np.shape(d_state):
            state_t = np.pad(state_t, pad_width=((0, 0), (1, 0)), mode='constant', constant_values=0)
        state_t1 = state_t + d_state

        return state_t1

