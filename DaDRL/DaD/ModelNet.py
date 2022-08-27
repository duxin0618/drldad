#
# Learner Wrapper helper classes for Data as Demonstrator (DaD) repository.
#
from sklearnex import patch_sklearn
patch_sklearn() #启动加速补丁
from DaDRL.DaD.dynamics import NormalDynamicModel
from DaDRL.DaD.radam import RAdam
import torch as th
from DaDRL.DaD.replay_buffer import AUXBuffer
from torch.distributions import Normal

import numpy as np

class DynamicsModel(object):

    def __init__(self, args, d_state, d_action, device):
        self.d_state = d_state
        self.d_action = d_action
        self.device = device
        self.clip_value = args.grad_clip
        self.args = args

        self.model = NormalDynamicModel(
            d_state=self.d_state, d_action=self.d_action, n_units=args.model_n_units, n_layers=args.model_n_layers,
            ensemble_size=args.model_ensemble_size, activation=args.model_activation,
            device=self.device
        )
        self.model_optimizer = RAdam(self.model.parameters(),
                                     lr=args.model_lr, weight_decay=args.model_weight_decay)


    def _update(self, states, actions, state_deltas):
        self.model_optimizer.zero_grad()

        loss = self.model.loss(states, actions, state_deltas)
        loss.backward()
        th.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_value)

        self.model_optimizer.step()
        return loss.item()

    def trainModel(self, buffer):
        # Training Dynamics Model
        # -------------------------------
        # if (self.args.model_training_freq is not None and self.args.model_training_n_batches > 0
        #     and self.step_i % self.args.model_training_freq == 0):
        loss = np.nan
        batch_i = 0
        while batch_i < self.args.model_training_n_batches:
            losses = []
            for states, actions, state_deltas in buffer.train_batches(self.args.model_ensemble_size, self.args.model_batch_size):
                th.set_grad_enabled(True)
                train_loss = self._update(states, actions, state_deltas)
                th.set_grad_enabled(False)
                losses.append(train_loss)
            batch_i += len(losses)
            loss = np.mean(losses)
        return loss

    def fit(self, state_t, action_t, state_t1):
        aux_buffer = AUXBuffer(d_state=self.d_state, d_action=self.d_action)
        aux_buffer.add(state_t, action_t, state_t1)
        self.trainModel(aux_buffer)

    # 返回下一状态
    def predict(self, state_t, action_t):
        # [emsable, batch_size, d_state]
        state = th.as_tensor(state_t, dtype=th.float32)
        action = th.as_tensor(action_t, dtype=th.float32)
        next_state = self.model.sample(state, action, sampling_type="ensemble")
        return next_state.cpu()



