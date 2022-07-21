

import numpy as np
import os
import scipy.io as sio
import time

def load_dataset(data_fname):
    if not os.path.isfile(data_fname):
        raise IOError('Cannot find file: {}'.format(data_fname))
    if os.path.splitext(data_fname)[1] == '.mat':
        data = sio.loadmat(data_fname)
    elif os.path.splitext(data_fname)[1] == '.npz':
        data = np.load(data_fname)
    else:
        raise Exception('Unknown input data extension')
    return data

def tensor_to_dataset(traj_tensor):

    # (50, 4, 25)
    # (50, 1, 25, 4)
    mat = np.vstack(traj_tensor.transpose((0,2,1)))  # 垂直方向进行堆叠
    return mat

def rms_error(trajs_a, trajs_b):

    def _rms_traj(traj1, traj2):
        """Helper function to compute the rms error between two trajectories. """
        err = (traj1- traj2)
        sq_err = err*err
        rms_err = np.sqrt(np.mean(np.sum(sq_err, axis=1)))
        return rms_err
    # If we havve more than one trajectory, compute the error across all the trajectories.
    if len(trajs_a.shape) == 3:

        rms = np.array([_rms_traj(trajs_a[:,:,n], trajs_b[:,:,n]) for n in range(trajs_a.shape[2])])
    else:
        rms = _rms_traj(trajs_a, trajs_b) 
    return rms

def ensure_2d(X, axis=1):
    if len(X.shape) == 1:
        X = np.expand_dims(X, axis=axis)
    elif len(X.shape) == 0:
        X = np.expand_dims(np.expand_dims(X, axis=axis), axis=axis)
    return X

def get_verboseprint(verbose):
    """This function is from: http://stackoverflow.com/a/5980173 """
    if verbose:
        def verboseprint(*args):
            # Print each argument separately so caller doesn't need to
            # stuff everything to be printed into a single string
            for arg in args:
               print(arg),
            print()
    else:   
        verboseprint = lambda *a: None      # do-nothing function
    return verboseprint
