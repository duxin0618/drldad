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
    env_name = ["InvertedDoublePendulum-v2",
                "CartPole-v0"][env_id]
    agent_class = [AgentA2C, AgentPPO, AgentDiscretePPO][drl_id]
    if env_name in {'CartPole-v0', 'CartPole-v1'}:
        env_func = gym.make
        env_args = {'env_num': 1,
                    'env_name': 'CartPole-v0',
                    'max_step': 200,
                    'state_dim': 4,
                    'action_dim': 2,
                    'if_discrete': True,
                    'target_return': 195,

                    'id': 'CartPole-v0'}
        args = Arguments(agent_class, env_func=env_func, env_args=env_args)

        args.target_step = args.max_step * 5
        args.reward_scale = 2 ** 0
        args.gamma = 0.99

        args.net_dim = 2 ** 6
        args.layer_num = 2
        args.batch_size = int(args.net_dim * 2)
        args.repeat_times = 2 ** 4
        args.break_step = 5e3
        args.if_allow_break = True # 到达break_step就停止
        args.eval_times = 2 ** 5
        args.eval_gap = 2 ** 1  # eva time sec
        args.lambda_h_term = 2 ** -5
    elif env_name in {"InvertedDoublePendulum-v2"}:
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
        args.target_step = args.max_step * 4
        args.eval_times = 2 ** 3
        args.layer_num = 3
        args.net_dim = 2 ** 6
        args.break_step = int(1e6)
        args.if_allow_break = False
        args.if_discrete = False
        args.eval_gap = 2 * 6
        args.eval_times = 2 ** 5
        args.learner_gpus = gpu_id
        args.random_seed += gpu_id
    else:
        raise ValueError('env_name:', env_name)


    threshold = 2.0
    args.useDaD = True
    args.useDaDTrain = True
    from DaDRL.static.cartpole import StaticFns as fc
    train_and_evaluate(args, threshold, fc)

if __name__ == '__main__':


    GPU_ID = 0
    DRL_ID = 2
    ENV_ID = 1

    demo_a2c_ppo(GPU_ID, DRL_ID, ENV_ID)
