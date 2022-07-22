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
    env_name = ["Pendulum-v0",
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

        args.target_step = args.max_step * 10
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
    elif env_name in {'Pendulum-v0', 'Pendulum-v1'}:
        from elegantrl.envs.CustomGymEnv import PendulumEnv
        env = PendulumEnv(env_name, target_return=-200)
        "TotalStep: 1e5, TargetReward: -200, UsedTime: 600s"
        args = Arguments(agent_class, env=env)
        args.reward_scale = 2 ** -1  # RewardRange: -1800 < -200 < -50 < 0
        args.gamma = 0.97
        args.target_step = args.max_step * 8
        args.eval_times = 2 ** 3
        args.layer_num = 2
        args.net_dim = 2 ** 6
        args.break_step = int(1e5)
        args.if_allow_break = True
        args.if_discrete = False
        args.eval_gap = 2 * 6
    else:
        raise ValueError('env_name:', env_name)


    threshold = 2.0
    args.useDaD = True
    args.useDaDTrain = True
    from DaDRL.static.pendulum import StaticFns as fc
    train_and_evaluate(args, threshold, fc)

if __name__ == '__main__':


    GPU_ID = 0
    DRL_ID = 1
    ENV_ID = 0

    demo_a2c_ppo(GPU_ID, DRL_ID, ENV_ID)
