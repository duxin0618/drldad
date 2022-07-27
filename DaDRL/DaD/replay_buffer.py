
import torch



class ReplayBufferList(list):  # for on-policy
    def __init__(self):
        list.__init__(self)

    def update_buffer(self, traj_list):
        cur_items = list(map(list, zip(*traj_list)))
        self[:] = [torch.cat(item, dim=0) for item in cur_items]

        steps = self[1].shape[0]
        r_exp = self[1].mean().item()
        return steps, r_exp

class DaDTrainBufferList(list):  # for dad training
    def __init__(self, max_size):
        list.__init__(self)
        self.max_size = max_size
        self.isFirst = True

    def update_buffer(self, traj_list):
        cur_items = list(map(list, zip(*traj_list)))
        self[:] = [torch.cat(item, dim=0) for item in cur_items]
        steps = self[1].shape[0]
        r_exp = self[1].mean().item()
        return steps, r_exp

    def augment_buffer(self, traj_list):
        if self.isFirst:
            self.isFirst = False
            self.update_buffer(traj_list)
        else:
            cur_items = list(map(list, zip(*traj_list)))

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


