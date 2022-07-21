import sys
import gym
import os


current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")
import gymenvs.cartpole_swingup_envs
from DaDRL.train.run import train_and_evaluate
from elegantrl.train.config import Arguments
from elegantrl.agents.AgentA2C import AgentA2C
from DaDRL.train.AgentPPO import AgentPPO, AgentDiscretePPO


def demo_a2c_ppo(gpu_id, drl_id, env_id):
    env_name = ['CartPoleSwingUpContinuous-v0',][env_id]
    agent_class = [AgentA2C, AgentPPO, AgentDiscretePPO][drl_id]
    if env_name in {'CartPoleSwingUpContinuous-v0'}:
        env_func = gym.make
        env_args = {'env_num': 1,
                    'env_name': 'CartPoleSwingUpContinuous-v0',
                    'max_step': 200,
                    'state_dim': 5,
                    'action_dim': 2,
                    'if_discrete': False,
                    'target_return': 600,

                    'id': 'CartPoleSwingUpContinuous-v0'}
        args = Arguments(agent_class, env_func=env_func, env_args=env_args)

        args.target_step = args.max_step * 2
        args.reward_scale = 2 ** 0
        args.gamma = 0.99

        args.net_dim = 2 ** 7
        args.layer_num = 2
        args.batch_size = int(args.net_dim * 2)
        args.repeat_times = 2 ** 4
        args.break_step = 7e5
        args.if_allow_break = False # 到达target_reward就停止
        args.eval_times = 2 ** 5
        args.eval_gap = 2 ** 8  # eva time sec
        args.lambda_h_term = 2 ** -5
        args.max_step = int(7e5)

    else:
        raise ValueError('env_name:', env_name)

    args.learner_gpus = gpu_id
    args.random_seed += gpu_id

    threshold = 2.0
    args.useDaD  = False
    args.useDaDTrain = False
    from DaDRL.static.ant_truncated_obs import StaticFns as fc
    train_and_evaluate(args, threshold, fc)



if __name__ == '__main__':
    # GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # >=0 means GPU ID, -1 means CPU
    # DRL_ID = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    # ENV_ID = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    # CartPole-v0
    # GPU_ID = 0
    # DRL_ID = 2
    # ENV_ID = 0


    # CartPoleSwingUpContinuous-v0
    GPU_ID = 2
    DRL_ID = 1
    ENV_ID = 0

    demo_a2c_ppo(GPU_ID, DRL_ID, ENV_ID)


