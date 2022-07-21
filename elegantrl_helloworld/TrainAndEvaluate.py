import sys
sys.path.append('../')
from elegantrl_helloworld.config import Arguments
from elegantrl_helloworld.run import train_agent, evaluate_agent
from elegantrl_helloworld.env import get_gym_env_args, PendulumEnv

def train_ppo_Reacher_v2(gpu_id=0):
    from elegantrl_helloworld.agent import AgentPPO
    agent_class = AgentPPO
    env_name = "Reacher-v2"

    import gym
    env = gym.make(env_name)
    env_func = gym.make
    env_args = get_gym_env_args(env, if_print=True)

    args = Arguments(agent_class, env_func, env_args)

    '''reward shaping'''
    args.gamma = 0.99
    args.reward_scale = 2 ** 0

    '''network update'''
    args.target_step = args.max_step * 8
    args.net_dim = 2 ** 6  # 128
    args.num_layer = 2
    args.batch_size = 2 ** 7
    args.repeat_times = 2 ** 4
    args.lambda_entropy = 0.04

    '''evaluate'''
    args.eval_gap = 2 ** 6
    args.eval_times = 2 ** 5
    args.break_step = int(1e6)

    args.learner_gpus = gpu_id
    train_agent(args)
    # evaluate_agent(args)
    print('| The cumulative returns of Reacher_-v2 is ∈ (-&&, -3.75+)')


def train_ppo_pendulum_v0(gpu_id=0):
    from elegantrl_helloworld.agent import AgentPPO
    agent_class = AgentPPO

    env = PendulumEnv()
    env_func = PendulumEnv
    env_args = get_gym_env_args(env, if_print=True)

    args = Arguments(agent_class, env_func, env_args)

    '''reward shaping'''
    args.reward_scale = 2 ** -1  # RewardRange: -1800 < -200 < -50 < 0
    args.gamma = 0.97

    '''network update'''
    args.target_step = args.max_step * 8  # ? 参数意义
    args.net_dim = 2 ** 6  # 128
    args.num_layer = 2
    args.batch_size = 2 ** 8  # 256
    args.repeat_times = 2 ** 5  # 32 ? 参数意义

    '''evaluate'''
    args.eval_gap = 2 ** 6
    args.eval_times = 2 ** 3
    args.break_step = int(8e5)

    args.learner_gpus = gpu_id
    train_agent(args)
    # evaluate_agent(args)
    print('| The cumulative returns of Pendulum-v0 is ∈ (-1600, (-1400, -200), 0)')


if __name__ == "__main__":
    train_ppo_pendulum_v0()
    # train_ppo_Reacher_v2()