import sys
import gym
import os

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")
from DaDRL.train.run import train_and_evaluate
from elegantrl.train.config import Arguments
from elegantrl.agents.AgentA2C import AgentA2C
from DaDRL.train.AgentPPO import AgentPPO, AgentDiscretePPO


def demo_a2c_ppo(gpu_id, drl_id, env_id):
    env_name = ['Swimmer-v2'][env_id]
    agent_class = [AgentA2C, AgentPPO, AgentDiscretePPO][drl_id]

    if env_name == 'Swimmer-v2':
        env_func = gym.make
        env_args = {'env_num': 1,
                    'env_name': 'Swimmer-v2',
                    'max_step': 1000,
                    'state_dim': 8,
                    'action_dim': 2,
                    'if_discrete': False,
                    'target_return': 360,
                    'id': 'Swimmer-v2'}
        args = Arguments(agent_class, env_func=env_func, env_args=env_args)

        args.target_step = args.max_step * 6
        args.reward_scale = 2 ** -2
        args.gamma = 0.99

        args.net_dim = 2 ** 6
        args.layer_num = 3
        args.batch_size = int(args.net_dim * 2)
        args.repeat_times = 2 ** 4

        args.break_step = int(3e6)
        args.if_allow_break = False

        args.eval_times = 2 ** 4
        args.eval_gap = 2 ** 6  # eva time sec
        args.lambda_h_term = 2 ** -5

    else:
        raise ValueError('env_name:', env_name)

    args.learner_gpus = gpu_id
    args.random_seed += gpu_id
    threshold = 0

    args.useDaD = False
    args.useDaDTrain = False
    args.if_state_expand = False

    n_k = 20  # traj number
    k_steps = 200  # traj length
    from DaDRL.static.walker2d import StaticFns as fc
    train_and_evaluate(args, threshold, fc, n_k, k_steps)



if __name__ == '__main__':

    # Swimmer-v2
    GPU_ID = 2
    DRL_ID = 1
    ENV_ID = 0

    demo_a2c_ppo(GPU_ID, DRL_ID, ENV_ID)



