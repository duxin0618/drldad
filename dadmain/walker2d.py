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
    env_name = ['Walker2d-v2'][env_id]
    agent_class = [AgentA2C, AgentPPO, AgentDiscretePPO][drl_id]
    if env_name == 'Walker2d-v2':
        from DaDRL.env.walker2d import Walker2dEnv
        env = Walker2dEnv(env_name, target_return=4000)
        args = Arguments(agent_class, env=env)

        args.eval_times = 2 ** 4
        args.reward_scale = 2 ** -4

        args.target_step = args.max_step * 6  # 6
        args.worker_num = 2
        args.eval_gap = 2 * 60
        args.net_dim = 2 ** 7
        args.layer_num = 3
        args.batch_size = int(args.net_dim * 2)
        args.repeat_times = 2 ** 4
        args.ratio_clip = 0.25
        args.gamma = 0.993
        args.lambda_entropy = 0.02
        args.lambda_h_term = 2 ** -5

        args.if_allow_break = False
        args.break_step = int(1e7)



    else:
        raise ValueError('env_name:', env_name)

    args.learner_gpus = gpu_id
    args.random_seed += gpu_id
    threshold = 0

    args.useDaD = True
    args.useDaDTrain = True
    args.if_state_expand = True

    n_k = 10  # traj number
    k_steps = 100  # traj length
    from DaDRL.static.walker2d import StaticFns as fc
    train_and_evaluate(args, threshold, fc, n_k, k_steps)



if __name__ == '__main__':

    # Walker2d-v2
    GPU_ID = 2
    DRL_ID = 1
    ENV_ID = 0

    demo_a2c_ppo(GPU_ID, DRL_ID, ENV_ID)



