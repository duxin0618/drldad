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
    env_name = ['CartPole-v0',
                'CartPole-v1',
                'Pendulum-v0',
                'Pendulum-v1',
                'LunarLanderContinuous-v2',
                'BipedalWalker-v3',
                'Hopper-v2',
                'Humanoid-v3',   # 7
                "InvertedDoublePendulum-v2"][env_id]
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

        args.target_step = args.max_step * 2
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

    elif env_name == 'LunarLanderContinuous-v2':

        env_func = gym.make
        env_args = {'env_num': 1,
                    'env_name': 'LunarLanderContinuous-v2',
                    'max_step': 1000,
                    'state_dim': 8,
                    'action_dim': 2,
                    'if_discrete': False,
                    'target_return': 270,

                    'id': 'LunarLanderContinuous-v2'}
        args = Arguments(agent_class, env_func=env_func, env_args=env_args)

        args.target_step = args.max_step * 2
        args.reward_scale = 2 ** -1
        args.gamma = 0.99

        args.net_dim = 2 ** 7
        args.layer_num = 3
        args.batch_size = int(args.net_dim * 2)
        args.repeat_times = 2 ** 4

        args.break_step = int(3e6)
        args.if_allow_break = False

        args.eval_times = 2 ** 2
        args.eval_gap = 2 ** 1  # eva time sec
        args.lambda_h_term = 2 ** -5
    elif env_name == 'BipedalWalker-v3':

        env_func = gym.make
        env_args = {'env_num': 1,
                    'env_name': 'BipedalWalker-v3',
                    'max_step': 1600,
                    'state_dim': 24,
                    'action_dim': 4,
                    'if_discrete': False,
                    'target_return': 300, }
        args = Arguments(agent_class, env_func=env_func, env_args=env_args)

        args.gamma = 0.98
        args.eval_times = 2 ** 4
        args.reward_scale = 2 ** -1

        args.target_step = args.max_step * 4
        args.worker_num = 2
        args.net_dim = 2 ** 7
        args.layer_num = 3
        args.batch_size = int(args.net_dim * 2)
        args.repeat_times = 2 ** 4
        args.ratio_clip = 0.25
        args.lambda_gae_adv = 0.96
        args.lambda_entropy = 0.02
        args.if_use_gae = True

        args.lambda_h_term = 2 ** -5
    elif env_name == 'Hopper-v2':

        env_func = gym.make
        env_args = {
            'env_num': 1,
            'env_name': 'Hopper-v2',
            'max_step': 1000,
            'state_dim': 11,
            'action_dim': 3,
            'if_discrete': False,
            'target_return': 3800.,
        }
        args = Arguments(agent_class, env_func=env_func, env_args=env_args)
        args.eval_times = 2 ** 2
        args.reward_scale = 2 ** -4

        args.target_step = args.max_step * 4  # 6
        args.worker_num = 2

        args.net_dim = 2 ** 7
        args.layer_num = 3
        args.batch_size = int(args.net_dim * 2)
        args.repeat_times = 2 ** 4
        args.ratio_clip = 0.25
        args.gamma = 0.993
        args.lambda_entropy = 0.02
        args.lambda_h_term = 2 ** -5

        args.if_allow_break = False
        args.break_step = int(8e6)
    elif env_name == 'Humanoid-v3':
        from elegantrl.envs.CustomGymEnv import HumanoidEnv
        env_func = HumanoidEnv
        env_args = {
            'env_num': 1,
            'env_name': 'Humanoid-v3',
            'max_step': 1000,
            'state_dim': 376,
            'action_dim': 17,
            'if_discrete': False,
            'target_return': 5000.,
        }
        args = Arguments(agent_class, env_func=env_func, env_args=env_args)
        args.reward_scale = 2 ** -4

        args.if_cri_target = False
        #
        # args.target_step = args.max_step * 16
        # args.lambda_entropy = 2 ** -6
        # args.worker_num = 2
        # args.batch_size = args.net_dim * 8
        # args.repeat_times = 2 ** 6

        args.learning_rate = 2 ** -14  # todo
        args.target_step = args.max_step * 8
        args.worker_num = 4
        args.batch_size = args.net_dim * 2
        args.repeat_times = 2 ** 5
        args.gamma = 0.995  # important
        args.if_use_gae = True
        args.coeff_gae_adv = 0.98 if gpu_id == 3 else 0.993  # todo
        args.coeff_entropy = 0.01

        args.eval_times = 2 ** 2
        args.max_step = int(8e7)
    elif env_name == 'Humanoid-v3.backup':
        from elegantrl.envs.CustomGymEnv import HumanoidEnv
        env_func = HumanoidEnv
        env_args = {
            'env_num': 1,
            'env_name': 'Humanoid-v3',
            'max_step': 1000,
            'state_dim': 376,
            'action_dim': 17,
            'if_discrete': False,
            'target_return': 5000.,
        }
        args = Arguments(agent_class, env_func=env_func, env_args=env_args)
        args.reward_scale = 2 ** -4

        args.if_cri_target = False
        #
        # args.target_step = args.max_step * 16
        # args.lambda_entropy = 2 ** -6
        # args.worker_num = 2
        # args.batch_size = args.net_dim * 8
        # args.repeat_times = 2 ** 6

        args.learning_rate = 2 ** -16
        args.target_step = args.max_step * 8
        args.worker_num = 4
        args.batch_size = args.net_dim * 2
        args.repeat_times = 2 ** 5
        args.gamma = 0.995  # important
        args.if_use_gae = True
        args.coeff_entropy = 0.01

        args.eval_times = 2 ** 1
        args.max_step = int(8e7)

    else:
        raise ValueError('env_name:', env_name)

    args.learner_gpus = gpu_id
    args.random_seed += gpu_id

    threshold = 2.0
    args.useDaD = False
    args.useDaDTrain = False
    from DaDRL.static.inverted_double_pendulum import StaticFns as fc
    train_and_evaluate(args, threshold, fc)



if __name__ == '__main__':
    # GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # >=0 means GPU ID, -1 means CPU
    # DRL_ID = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    # ENV_ID = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    # CartPole-v0
    # GPU_ID = 0
    # DRL_ID = 2
    # ENV_ID = 0


    # Pendulum-v0
    GPU_ID = 0
    DRL_ID = 1
    ENV_ID = 2

    demo_a2c_ppo(GPU_ID, DRL_ID, ENV_ID)

    #no 519.5576710700989
    #


