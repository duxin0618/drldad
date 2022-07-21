# # import gym
# import torch
# from torch.distributions import Normal
# # env = gym.make('CartPole-v0')
# # print(dir(env))
# # print(repr(env.class_name))
# # print(gym.logger.set_level(40))
# a = torch.tensor([[2,2,2,2],[4,4,4,4]], dtype=torch.float32)
# dist = Normal(0, 1).log_prob(a).sum(-1, keepdim=True)
# print(a.mean())
# # tensor([-1.0439, -1.0439, -1.0439, -1.0439])
# import numpy as np
# t = np.load("E:\\WorkSpace\\pylab\\IML\\ElegantRL-master\\elegantrl_helloworld\\BipedalWalker-v3_PPO_0\\recorder.npy")
# print(t)
#
# import gym
# env = gym.make("Pendulum-v0")
# print(env.env.unwrapped.__str__)


from sklearn.linear_model import Ridge


ridge = Ridge(alpha=1e-4, fit_intercept=True)
print(ridge.get_params())
print(ridge.__dict__['coef_'])