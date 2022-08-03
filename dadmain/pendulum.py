import sys
import gym
import os

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")
from DaDRL.train.run import train_and_evaluate
from DaDRL.train.config import Arguments
from elegantrl.agents.AgentA2C import AgentA2C
from DaDRL.train.AgentPPO import AgentPPO, AgentDiscretePPO


def demo_a2c_ppo(gpu_id, drl_id, env_id):
    env_name = ["Pendulum-v0"][env_id]
    agent_class = [AgentA2C, AgentPPO, AgentDiscretePPO][drl_id]
    if env_name in {'Pendulum-v0', 'Pendulum-v1'}:
        from DaDRL.env.pendulum import PendulumEnv
        env = PendulumEnv(env_name, target_return=-200)
        args = Arguments(agent_class, env=env)

        args.reward_scale = 2 ** -1  # RewardRange: -1800 < -200 < -50 < 0
        args.gamma = 0.97
        args.target_step = args.max_step * 6
        args.eval_times = 2 ** 3
        args.layer_num = 2
        args.net_dim = 2 ** 6
        args.break_step = int(1e5)
        args.if_allow_break = False
        args.if_discrete = False
        args.eval_gap = 2 * 6
    else:
        raise ValueError('env_name:', env_name)

    args.learner_gpus = gpu_id
    args.random_seed += gpu_id

    threshold = 2.0
    args.useDaD = True
    args.useDaDTrain = True
    args.if_state_expand = False
    n_k = 40             # traj number
    k_steps = 10         # traj length
    args.n_k = n_k
    args.k_steps = k_steps
    from DaDRL.static.pendulum import StaticFns
    train_and_evaluate(args, threshold, StaticFns)

if __name__ == '__main__':


    GPU_ID = 1
    DRL_ID = 1
    ENV_ID = 0

    demo_a2c_ppo(GPU_ID, DRL_ID, ENV_ID)
