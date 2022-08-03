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
    env_name = ['Humanoid-v3'][env_id]
    agent_class = [AgentA2C, AgentPPO, AgentDiscretePPO][drl_id]
    if env_name == 'Humanoid-v3':
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

        args.net_dim = 2 ** 7
        args.layer_num = 5
        args.learning_rate = 2 ** -14  # todo
        args.target_step = args.max_step * 8
        args.worker_num = 4
        args.batch_size = args.net_dim * 2
        args.repeat_times = 2 ** 6
        args.gamma = 0.995  # important
        args.if_use_gae = True
        args.coeff_gae_adv = 0.98 if gpu_id == 3 else 0.993  # todo
        args.coeff_entropy = 0.01

        args.eval_times = 2 ** 5
        args.max_step = int(8e7)
        args.if_allow_break = False
        args.eval_gap = 15 * 60

    else:
        raise ValueError('env_name:', env_name)

    args.learner_gpus = gpu_id
    args.random_seed += gpu_id

    threshold = 2.0
    args.useDaD = False
    args.useDaDTrain = False
    args.if_state_expand = False

    n_k = 6  # traj number
    k_steps = 10  # traj length

    # random
    from DaDRL.static.Reacher import StaticFns as fc
    train_and_evaluate(args, threshold, fc, n_k, k_steps)


if __name__ == '__main__':
    # Humanoid-v3
    GPU_ID = 3
    DRL_ID = 1
    ENV_ID = 0

    demo_a2c_ppo(GPU_ID, DRL_ID, ENV_ID)



