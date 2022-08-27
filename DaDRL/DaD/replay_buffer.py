
import torch
import math
import numpy as np

class ReplayBufferList(list):  # for on-policy
    def __init__(self):
        list.__init__(self)

    def update_buffer(self, traj_list):
        cur_items = list(map(list, zip(*traj_list)))
        self[:] = [torch.cat(item, dim=0) for item in cur_items]

        steps = self[1].shape[0]
        r_exp = self[1].mean().item()
        return steps, r_exp

class ModelTrainBufferList(list):  # for dad training
    def __init__(self, max_size):
        list.__init__(self)
        self.max_size = math.inf
        self.isFirst = True

    def update_buffer(self, traj_list, is_map):
        if not is_map:
            cur_items = list(map(list, zip(*traj_list)))
        else:
            cur_items = traj_list
        self[:] = [torch.cat(item, dim=0) for item in cur_items]
        steps = self[1].shape[0]
        r_exp = self[1].mean().item()
        return steps, r_exp

    def augment_buffer(self, traj_list, is_map=False):
        if self.isFirst:
            self.isFirst = False
            self.update_buffer(traj_list, is_map)
        else:
            if not is_map:
                cur_items = list(map(list, zip(*traj_list)))
            else:
                cur_items = traj_list
            length = len(cur_items)
            for idx in range(length):
                self[idx] = torch.cat([self[idx], cur_items[idx][0]], dim=0)
            self.remove_buffer()
            steps = self[1].shape[0]
            r_exp = self[1].mean().item()
            return steps, r_exp

    def remove_buffer(self):
        length = len(self[:])
        for idx in range(length):
            if len(self[idx]) > self.max_size:
                cur_index = len(self[idx]) - self.max_size
                self[idx] = self[idx][cur_index: ]


class Buffer:
    def __init__(self, d_state, d_action, size):
        """
        data buffer that holds transitions

        Args:
            d_state: dimensionality of state
            d_action: dimensionality of action
            size: maximum number of transitions to be stored (memory allocated at init)
        """
        # Dimensions
        self.size = size
        self.d_state = d_state
        self.d_action = d_action

        # Main Attributes
        self.states = torch.zeros(size, d_state).float()
        self.actions = torch.zeros(size, d_action).float()
        self.state_deltas = torch.zeros(size, d_state).float()
        self.rewards = torch.zeros(size, 1).float()

        # Other attributes
        self.normalizer = None
        self.ptr = 0
        self.is_full = False

    def setup_normalizer(self, normalizer):
        self.normalizer = normalizer

    def _add(self, buffer, arr):
        n = arr.size(0)
        excess = self.ptr + n - self.size  # by how many elements we exceed the size
        if excess <= 0:  # all elements fit
            a, b = n, 0
        else:
            a, b = n - excess, excess  # we need to split into a + b = n; a at the end and the rest in the beginning
        buffer[self.ptr:self.ptr + a] = arr[:a]
        buffer[:b] = arr[a:]

    def add(self, states, actions, next_states, rewards):
        """
        add transition(s) to the buffer

        Args:
            states: pytorch Tensors of (n_transitions, d_state) shape
            actions: pytorch Tensors of (n_transitions, d_action) shape
            next_states: pytorch Tensors of (n_transitions, d_state) shape
        """
        states, actions, next_states, rewards = [x.clone().cpu() for x in [states, actions, next_states, rewards]]

        state_deltas = next_states - states
        n_transitions = states.size(0)

        assert n_transitions <= self.size

        self._add(self.states, states)
        self._add(self.actions, actions)
        self._add(self.state_deltas, state_deltas)
        self._add(self.rewards, rewards)

        if self.ptr + n_transitions >= self.size:
            self.is_full = True

        self.ptr = (self.ptr + n_transitions) % self.size

        if self.normalizer is not None:
            for s, a, ns, r in zip(states, actions, state_deltas, rewards):
                self.normalizer.add(s, a, ns, r)

    def view(self):
        n = len(self)

        s = self.states[:n]
        a = self.actions[:n]
        s_delta = self.state_deltas[:n]
        ns = s + s_delta

        return s, a, ns, s_delta

    def train_batches(self, ensemble_size, batch_size):
        """
        return an iterator of batches

        Args:
            batch_size: number of samples to be returned
            ensemble_size: size of the ensemble

        Returns:
            state of size (ensemble_size, n_samples, d_state)
            action of size (ensemble_size, n_samples, d_action)
            next state of size (ensemble_size, n_samples, d_state)
        """
        num = len(self)
        indices = [np.random.permutation(range(num)) for _ in range(ensemble_size)]
        indices = np.stack(indices)

        for i in range(0, num, batch_size):
            j = min(num, i + batch_size)

            if (j - i) < batch_size and i != 0:
                # drop the last incomplete batch
                return

            batch_size = j - i

            batch_indices = indices[:, i:j]
            batch_indices = batch_indices.flatten()

            states = self.states[batch_indices]
            actions = self.actions[batch_indices]
            state_deltas = self.state_deltas[batch_indices]

            states = states.reshape(ensemble_size, batch_size, self.d_state)
            actions = actions.reshape(ensemble_size, batch_size, self.d_action)
            state_deltas = state_deltas.reshape(ensemble_size, batch_size, self.d_state)

            yield states, actions, state_deltas

    def sample(self, batch_size, device):
        """
        This function will only sample the data with size batch_size.

        Args:
            batch_size: number of samples to be returned
            device: torch.Device

        Returns:
            state of size (n_samples, d_state)
            action of size (n_samples, d_action)
            next state of size (n_samples, d_state)
            reward of size (n_samples, 1)
        """
        curr_size = len(self)
        sample_size = min(curr_size, batch_size)
        indices = np.random.randint(0, curr_size, sample_size)

        states = self.states[indices].reshape(sample_size, self.d_state).to(device)
        actions = self.actions[indices].reshape(sample_size, self.d_action).to(device)
        state_deltas = self.state_deltas[indices].reshape(sample_size, self.d_state).to(device)
        rewards = self.rewards[indices].reshape(sample_size, 1).to(device)

        return states, actions, states + state_deltas, rewards

    def trajectory(self, if_state_expand):
        s, a, n_detla, ns = self.view()
        aux_len = len(self)
        max_train_len = 256
        train_sample = (aux_len - aux_len % max_train_len)
        it_traj = train_sample / max_train_len

        if if_state_expand:
            r_state_dim = self.d_state + 1
        else:
            r_state_dim = self.d_state

        states = s[:train_sample, :].reshape(int(it_traj), int(max_train_len), r_state_dim)[:, :-1]
        actions = a[:train_sample, :].reshape(int(it_traj), int(max_train_len), self.d_action)[:, :-1]
        states_next = s[:train_sample, :].reshape(int(it_traj), int(max_train_len), r_state_dim)[:, 1:]
        return states, actions, states_next

    def __len__(self):
        return self.size if self.is_full else self.ptr

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

        # backward compatibility with old buffers
        if 'size' not in state and 'ptr' not in state and 'is_full' not in state:
            self.size = state['buffer_size']
            self.ptr = state['_n_elements'] % state['buffer_size']
            self.is_full = (state['_n_elements'] > state['buffer_size'])
            del self.buffer_size
            del self._n_elements
            del self.ensemble_size

class AUXBuffer:
    def __init__(self, d_state, d_action):

        self.d_state = d_state
        self.d_action = d_action

        # Other attributes
        self.normalizer = None
        self.ptr = 0
        self.is_full = False

    def setup_normalizer(self, normalizer):
        self.normalizer = normalizer

    def _add(self, buffer, arr):
        n = arr.size(0)
        excess = self.ptr + n - self.size  # by how many elements we exceed the size
        if excess <= 0:  # all elements fit
            a, b = n, 0
        else:
            a, b = n - excess, excess  # we need to split into a + b = n; a at the end and the rest in the beginning
        buffer[self.ptr:self.ptr + a] = arr[:a]
        buffer[:b] = arr[a:]

    def add(self, states, actions, next_states):
        """
        add transition(s) to the buffer

        Args:
            states: pytorch Tensors of (n_transitions, d_state) shape
            actions: pytorch Tensors of (n_transitions, d_action) shape
            next_states: pytorch Tensors of (n_transitions, d_state) shape
        """
        # Main Attributes
        size = states.shape[0] * states.shape[2]
        self.states = torch.zeros(size, self.d_state)
        self.actions = torch.zeros(size, self.d_action)
        self.state_deltas = torch.zeros(size, self.d_state)

        states = np.vstack(states)
        actions = np.vstack(actions)
        next_states = np.vstack(next_states)

        state_deltas = next_states - states
        n_transitions = states.shape[0]

        self._add(self.states, states)
        self._add(self.actions, actions)
        self._add(self.state_deltas, state_deltas)

        if self.ptr + n_transitions >= self.size:
            self.is_full = True

        self.ptr = (self.ptr + n_transitions) % self.size

        if self.normalizer is not None:
            for s, a, ns in zip(states, actions, state_deltas):
                self.normalizer.add(s, a, ns)

    def view(self):
        n = len(self)

        s = self.states[:n]
        a = self.actions[:n]
        s_delta = self.state_deltas[:n]
        ns = s + s_delta

        return s, a, ns, s_delta

    def train_batches(self, ensemble_size, batch_size):
        """
        return an iterator of batches

        Args:
            batch_size: number of samples to be returned
            ensemble_size: size of the ensemble

        Returns:
            state of size (ensemble_size, n_samples, d_state)
            action of size (ensemble_size, n_samples, d_action)
            next state of size (ensemble_size, n_samples, d_state)
        """
        num = len(self)
        indices = [np.random.permutation(range(num)) for _ in range(ensemble_size)]
        indices = np.stack(indices)

        for i in range(0, num, batch_size):
            j = min(num, i + batch_size)

            if (j - i) < batch_size and i != 0:
                # drop the last incomplete batch
                return

            batch_size = j - i

            batch_indices = indices[:, i:j]
            batch_indices = batch_indices.flatten()

            states = self.states[batch_indices]
            actions = self.actions[batch_indices]
            state_deltas = self.state_deltas[batch_indices]

            states = states.reshape(ensemble_size, batch_size, self.d_state)
            actions = actions.reshape(ensemble_size, batch_size, self.d_action)
            state_deltas = state_deltas.reshape(ensemble_size, batch_size, self.d_state)

            yield states, actions, state_deltas

    def sample(self, batch_size, device):
        """
        This function will only sample the data with size batch_size.

        Args:
            batch_size: number of samples to be returned
            device: torch.Device

        Returns:
            state of size (n_samples, d_state)
            action of size (n_samples, d_action)
            next state of size (n_samples, d_state)
            reward of size (n_samples, 1)
        """
        curr_size = len(self)
        sample_size = min(curr_size, batch_size)
        indices = np.random.randint(0, curr_size, sample_size)

        states = self.states[indices].reshape(sample_size, self.d_state).to(device)
        actions = self.actions[indices].reshape(sample_size, self.d_action).to(device)
        state_deltas = self.state_deltas[indices].reshape(sample_size, self.d_state).to(device)
        rewards = self.rewards[indices].reshape(sample_size, 1).to(device)

        return states, actions, states + state_deltas, rewards

    def trajectory(self, if_state_expand):
        s, a, n_detla, ns = self.view()
        aux_len = len(self)
        max_train_len = 256
        train_sample = (aux_len - aux_len % max_train_len)
        it_traj = train_sample / max_train_len

        if if_state_expand:
            r_state_dim = self.d_state + 1
        else:
            r_state_dim = self.d_state

        states = s[:train_sample, :].reshape(int(it_traj), int(max_train_len), r_state_dim)[:, :-1]
        actions = a[:train_sample, :].reshape(int(it_traj), int(max_train_len), self.d_action)[:, :-1]
        states_next = s[:train_sample, :].reshape(int(it_traj), int(max_train_len), r_state_dim)[:, 1:]
        return np.array(states), np.array(actions), np.array(states_next)

    def __len__(self):
        return self.size if self.is_full else self.ptr

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)

        # backward compatibility with old buffers
        if 'size' not in state and 'ptr' not in state and 'is_full' not in state:
            self.size = state['buffer_size']
            self.ptr = state['_n_elements'] % state['buffer_size']
            self.is_full = (state['_n_elements'] > state['buffer_size'])
            del self.buffer_size
            del self._n_elements
            del self.ensemble_size

