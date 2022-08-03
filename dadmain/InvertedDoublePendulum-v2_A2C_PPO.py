import sys
import gym
import os

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")
from DaDRL.train.run import train_and_evaluate
from elegantrl.train.config import Arguments
from elegantrl.agents.AgentA2C import AgentA2C
# from elegantrl.agents.AgentPPO import AgentDiscretePPO
from DaDRL.train.AgentPPO import AgentPPO, AgentDiscretePPO


def demo_a2c_ppo(gpu_id, drl_id, env_id):
    env_name = ["InvertedDoublePendulum-v2"][env_id]
    agent_class = [AgentA2C, AgentPPO, AgentDiscretePPO][drl_id]
    if env_name in {"InvertedDoublePendulum-v2"}:
        env_func = gym.make
        env_args = {'env_num': 1,
                    'env_name': 'InvertedDoublePendulum-v2',
                    'max_step': 1000,
                    'state_dim': 11,
                    'action_dim': 1,
                    'if_discrete': False,
                    'target_return': 9100,
                    'id': 'InvertedDoublePendulum-v2'}
        args = Arguments(agent_class, env_func=env_func, env_args=env_args)

        args.reward_scale = 2 ** 0
        args.gamma = 0.97
        args.target_step = args.max_step * 6
        args.eval_times = 2 ** 3
        args.layer_num = 3
        args.net_dim = 2 ** 6
        args.break_step = int(2e5)
        args.if_allow_break = False
        args.if_discrete = False
        args.eval_gap = 2 * 6
        args.eval_times = 2 ** 5
        args.learner_gpus = gpu_id
        args.random_seed += gpu_id
    else:
        raise ValueError('env_name:', env_name)


    threshold = 2.0
    args.useDaD = False
    args.useDaDTrain = False
    args.if_state_expand = False
    n_k = 20  # traj number
    k_steps = 200  # traj length
    from DaDRL.static.inverted_double_pendulum import StaticFns as fc
    train_and_evaluate(args, threshold, fc, n_k, k_steps)

if __name__ == '__main__':


    GPU_ID = 1
    DRL_ID = 2
    ENV_ID = 0

    demo_a2c_ppo(GPU_ID, DRL_ID, ENV_ID)
