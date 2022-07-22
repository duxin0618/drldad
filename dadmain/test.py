# # # import torch
# # # # traj_list = [
# # # #             [list() for _ in range(4 if True else 5)]
# # # #             for _ in range(1)
# # # #         ]
# # # # print(traj_list)
# # # traj_list = []
# # # tensor_state = [1,2]
# # # reward = 1
# # # done = 2
# # # tensor_action = 3
# # # tensor_noise = 4
# # # for _ in range(4):
# # #     traj_list.append((tensor_state, reward, done, tensor_action, tensor_noise))
# # #
# # #
# # # print(traj_list)
# # # # [([1, 2], 1, 2, 3, 4), ([1, 2], 1, 2, 3, 4), ([1, 2], 1, 2, 3, 4), ([1, 2], 1, 2, 3, 4)]
# # # cur_items = list(map(list, zip(*traj_list)))
# # #
# # #
# # # #
# # # # steps = self[1].shape[0]
# # # # r_exp = self[1].mean().item()
# # # print(cur_items)
# # # # [[[1, 2], [1, 2], [1, 2], [1, 2]], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]
# # # print([torch.cat(item, dim=0) for item in cur_items])
# # # class a:
# # #     def __init__(self):
# # #         self.states = None
# # #
# # #     def p(self):
# # #         print(self.states[0])
# # #
# # # a().p()
# # # import numpy as np
# # # a = [1,2,3]
# # # b = np.expand_dims(a, axis=0)
# # # print(b)
# #
# # import matplotlib.pyplot as plt
# # import numpy as np
# # steps = range(3)
# # r_exp = np.array([1,2,3])
# # r_std = 0.1
# #
# # color = "royalblue"
# #
# # plt.title("explore_rewards", fontweight ="bold")
# # plt.ylabel("Episode Return")
# # plt.xlabel("Episode Steps")
# # plt.plot(steps, r_exp, label="Episode Return", color=color)
# # plt.fill_between(steps, r_exp - r_std, r_exp + r_std, facecolor=color, alpha=0.3)
# # plt.grid()
# # # plt.savefig("./result/Pendulum-v0_PPO_0/explore_rewards.jpg")
# # plt.show()
# # # import numpy as np
# # # a = [1,2,3] ; b = [1,2,3]
# # # c = list()
# # # c.append(a)
# # # c.append(b)
# # # d = np.array(c).ravel()
# # #
# # # print(d.size)
#
#
# import numpy as np
# def plot_info(title, y1: np.array, cwd, y2: np.array=None):
#     # delete Abnormal point
#     def delete_no_data(data_array)->np.array:
#         mean = np.mean(data_array)
#         std = np.std(data_array)
#         print(std)
#         preprocessed_data_array = [x for x in data_array if (x > mean - std)]
#         preprocessed_data_array = [x for x in preprocessed_data_array if (x < mean + std)]
#         return preprocessed_data_array
#
#     import matplotlib.pyplot as plt
#
#     y1 = y1.ravel()
#     y1mean = np.mean(delete_no_data(y1))
#     steps = y1.size
#     length = range(steps)
#
#     filename = "dad error"
#     title = title
#     plt.title(title, fontweight="bold")
#     plt.ylabel("dad error")
#     plt.xlabel("Episode")
#     plt.plot(length, y1, marker="^", linestyle="-", color="r", label="minerror -mean: "+str(y1mean))
#     if y2 is not None:
#         y2 = y2.ravel()
#         y2mean = np.mean(y2)
#         plt.plot(length, y2, marker="s", linestyle="-", color="b", label="initerror -mean: "+str(y2mean))
#
#     plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.98))
#     plt.grid()
#     plt.savefig(f"{cwd}/{filename}.jpg")
#     plt.show()
#
#
#
#
# # a = [1,2,3,4,5,6]
# # b = [2,3,4,5,6,7]
# # a = np.asarray(a)
# # b = np.asarray(b)
# # plot_info("error", a, "./", b)
#
# import numpy as np
#
# # a = [1,2,3,4]
# # b = [2,3,4,5]
# # #
# # # l = list()
# # # l.append(a)
# # # l.append(b)
# #
# # d = dict()
# # d["mer"] = a
# # d["ime"] = b
# # np.save("a", d)
# # c = np.load("a.npy", allow_pickle=True)
# #
# # print(c.item()['mer'])
# # print(type(c))
#
# def read_npy_data(cwd):
#     data = np.load(cwd, allow_pickle=True)
#     a = data.item()['min_error']
#     b = data.item()['init_error']
#
#     # print(data.item()['min_error'])
#     # print(data.item()['init_error'])
#     plot_info("model error", a, "./", b)
# # def store_npy_data(*data, cwd, name):
# #     cur = dict()
# #     for key, da in data:
# #         cur[key] = da
# #     np.save(cwd+"/"+name, cur)
# #
# # a = [1,2]
# #
# # store_npy_data(("3111", a) ,("3112", a),("3113", a), cwd="./", name="a")
# read_npy_data("./dad model.npy")

# save explore rewards

# import numpy as np
# def explorerewards_curve_plot(cwd, rewards, useDad):
#     import matplotlib.pyplot as plt
#     rewards = np.array(rewards).ravel()
#     steps = rewards.size
#     length = range(steps)
#     r_std = 0.1
#
#     color = "royalblue"
#     if useDad:
#         title = "Use DaD Explore Episode Return"
#         filename = "Use_DaD_explore_rewards"
#     else:
#         filename = "No_DaD_explore_rewards"
#         title = "No DaD Explore Episode Return"
#     plt.title(title, fontweight="bold")
#     plt.ylabel("Episode Return")
#     plt.xlabel("Episode Steps")
#     plt.plot(length, rewards, label="Explore Episode Return", color=color)
#     plt.fill_between(length, rewards - r_std, rewards + r_std, facecolor=color, alpha=0.3)
#     plt.grid()
#     # plt.savefig(f"{cwd}/{filename}.jpg")
#     plt.show()
#
#
# rewards = np.load("./explore_rewards.npy", allow_pickle=True)
# print(rewards.item()['explore_rewards'])
# explorerewards_curve_plot("./", rewards.item()['explore_rewards'], True)
#
# #
# import torch
# t11 = torch.asarray([[1,2,3],[1,2,3],[1,2,3]])
# t12 = torch.asarray([[2],[2],[2]])
# a = [t11, t12]
# t21 = torch.asarray([[2,3,4],[2,3,4],[2,3,4]])
# t22 = torch.asarray([[3],[3],[3]])
# b = [t21, t22]
#
# # c = []
# # for idx in range(2):
# #     c.append(torch.cat((a[idx], b[idx])))
# c = [torch.cat((a[idx], b[idx])) for idx in range(2)]
# print(c)
# import torch
# a = torch.tensor([0.97])
# # print(torch.unsqueeze(a, 0))
# print(torch.round(a))
# import numpy as np
# a = [1,2,3]
# a = np.asarray(a)
# print(type(a))
# print(isinstance(a, np.ndarray))

##################store###################################################
# def train_and_evaluate(args):
#     torch.set_grad_enabled(False)
#     args.init_before_training()
#     gpu_id = args.learner_gpus
#
#     """init"""
#     env = build_env(args.env, args.env_func, args.env_args)
#
#     agent = init_agent(args, gpu_id, env)
#     buffer = init_buffer(args, gpu_id)
#
#     evaluator = init_evaluator(args, gpu_id)
#
#     """ DaD"""
#     dad = Dagger((20, 10), args)
#     dad_buffer = init_dad_buffer()
#     dad_train_buffer = init_dad_trainbuffer()
#
#     if_save = None
#
#     agent.state = env.reset()
#
#     if args.if_off_policy:
#         trajectory = agent.explore_env(env, args.target_step)
#         buffer.update_buffer((trajectory,))
#
#     """start training"""
#     cwd = args.cwd
#     break_step = args.break_step  # 为replaybuffer准备的
#     target_step = args.target_step
#     if_allow_break = args.if_allow_break
#     del args
#
#     explore_rewards = list()
#
#     """ record dad model error"""
#     init_model_errors = list()
#     min_model_errors = list()
#     threshold = 4.0
#     if_train = True
#     while if_train:
#         trajectory, explore_reward = agent.explore_env(env, target_step)
#         steps, r_exp = buffer.update_buffer((trajectory,))
#
#         explore_rewards.extend(explore_reward) # explore rewards
#         torch.set_grad_enabled(True)
#         logging_tuple = agent.update_net(buffer)
#         torch.set_grad_enabled(False)
#
#         '''
#         DaD train
#         '''
#
#         useDad = True
#         useDadTrain = True
#         if useDad:
#             dad_train_buffer.update_buffer((trajectory,))
#             min_model_error, init_model_error = dad.train(dad_train_buffer)
#             init_model_errors.append(init_model_error)
#             min_model_errors.append(min_model_error)
#
#             print("dad_error : ", min_model_error)
#
#             if min_model_error <= threshold and useDadTrain:
#                 dad_trajectory = dad.explore_env(env, target_step, dad_train_buffer)
#                 dad_steps, dad_r_exp = dad_buffer.update_buffer((dad_trajectory,))
#
#                 torch.set_grad_enabled(True)
#                 dad_logging_tuple = agent.update_net(dad_buffer)
#                 torch.set_grad_enabled(False)
#
#         '''
#         The End
#         '''
#         (if_reach_goal, if_save) = evaluator.evaluate_save_and_plot(
#             agent.act, steps, r_exp, logging_tuple
#         )
#         dont_break = not if_allow_break
#         not_reached_goal = not if_reach_goal
#         stop_dir_absent = not os.path.exists(f"{cwd}/stop")
#         if_train = (
#             (dont_break or not_reached_goal)
#             and evaluator.total_step <= break_step
#             and stop_dir_absent
#         )
#     print(f"| UsedTime: {time.time() - evaluator.start_time:.0f} | SavedDir: {cwd}")
#     agent.save_or_load_agent(cwd, if_save=if_save)
#     buffer.save_or_load_history(cwd, if_save=True) if agent.if_off_policy else None
#
#     evaluator.save_explorerewards_curve_plot_and_npy(cwd, explore_rewards, useDad, threshold)
#     if useDad:
#         init_model_errors = np.asarray(init_model_errors)
#         min_model_errors = np.asarray(min_model_errors)
#
#         """draw min_error and init_error curve"""
#         plot_info("Dad model Init_error and Min_error", init_model_errors, "init_model_errors", cwd, "dad error",
#                   min_model_errors, "min_model_errors")
#
#         """store min_error and init_error data"""
#         store_npy_data(("min_error", min_model_errors), ("init_error", init_model_errors), cwd=cwd, name="dad model error")

# env = gym.make("Hopper-v2")
# print(env.action_space)
# print(env.observation_space)
# Walker2d-v2
# Swimmer-v2
# import gym
# import gymenvs.cartpole_swingup_envs
# env = gym.make("CartPoleSwingUpDiscrete-v0")
# print(env.action_space)
# print(env.observation_space)
# print(env.action_space.sample())
#
import torch
#
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())

# from gym import envs
#
# all_envs = envs.registry.all()
# env_ids = [env_spec.id for env_spec in all_envs]
#
# for env_id in env_ids:
#     print(env_id)
# import numpy as np
#
# a = np.array([1])
# b = torch.tensor(2)
# a[0] = b
# print(type(a))
#
# import gym
# import gymenvs.cartpole_swingup_envs

# Could be one of:
# CartPoleSwingUp-v0, CartPoleSwingUp-v1
# If you have PyTorch installed:
# TorchCartPoleSwingUp-v0, TorchCartPoleSwingUp-v1
# env = gym.make("CartPoleSwingUpContinuous-v0")
# done = False
# env.reset()
# while not done:
#     action = env.action_space.sample()
#     obs, rew, done, info = env.step(action)
#     print(rew)
#     env.render()

# import metaworld
# import random
#
# print(metaworld.ML1.ENV_NAMES)  # Check out the available environments
#
# ml1 = metaworld.ML1('pick-place-v2') # Construct the benchmark, sampling tasks
#
# env = ml1.train_classes['pick-place-v2']()  # Create an environment with task `pick_place`
# task = random.choice(ml1.train_tasks)
# env.set_task(task)  # Set task
#
# obs = env.reset()  # Reset environment
# a = env.action_space.sample()  # Sample an action
# obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
# print(env.action_space)
# print(len(obs))
# print(reward)
# class ReplayBufferList(list):  # for on-policy
#     def __init__(self, max_size):
#         list.__init__(self)
#         self.max_size = max_size
#
#     def update_buffer(self, traj_list):
#         cur_items = list(map(list, zip(*traj_list)))
#         self[:] = [torch.cat(item, dim=0) for item in cur_items]
#         steps = self[1].shape[0]
#         r_exp = self[1].mean().item()
#         return steps, r_exp
#
#     def augment_buffer(self, traj_list):
#         cur_items = list(map(list, zip(*traj_list)))
#
#         length = len(cur_items)
#         for idx in range(length):
#             self[idx] = torch.cat([cur_items[idx][0], self[idx]], dim=0)
#         self.remove_buffer()
#         steps = self[1].shape[0]
#         r_exp = self[1].mean().item()
#         return steps, r_exp
#
#     def remove_buffer(self):
#         length = len(self[:])
#         for idx in range(length):
#             if len(self[idx]) > 8:
#                 cur_index = len(self[idx]) - 8
#                 self[idx] = self[idx][cur_index: ]
#
# states = []
# actions = []
# def a(max_size):
#     rb = ReplayBufferList(max_size=max_size)
#     return rb
# rb = a(1000)
# for i in range(10):
#     states.append([-0.9862, -0.1654,  0.5984])
#     actions.append([-0.1100])
# traj = [torch.tensor(states), torch.tensor(actions)]
# print(rb.update_buffer(traj_list=(traj,)))
# print(rb.remove_buffer())



# import gym
# env = gym.make("InvertedDoublePendulum-v2")
# env.seed(0)
# print("state: ", env.reset())
# print("xpos: ", env.sim.data.site_xpos[:])
# print("qpos: ", env.sim.data.qpos[:])
# """
# state:  [-0.08912799  0.09294385  0.02653507  0.99567135  0.99964788 -0.05766308
#  -0.03045979 -0.09994328  0.          0.          0.        ]
# xpos:  [[0.03823712 0.         1.1931155 ]]
# qpos:  [-0.08912799  0.09307819  0.02653819]
# """

import gym
# env = gym.make("Walker2d-v2")
# env.seed(0)
# print("state: ", env.reset())
# print("qpos: ", env.sim.data.qpos[:])

# print(env.sim.data)
"""
state:  [ 1.25465391e+00  1.32690946e-03 -2.09982656e-03 -3.97515743e-03
  1.73076348e-03 -1.07423260e-03  1.69846067e-03 -4.40161031e-03
  2.69872445e-04 -4.59709307e-03  4.77944075e-03 -3.70632589e-04
  1.29647936e-03  4.47048431e-03  1.99277824e-03 -4.65296951e-04
  4.02781858e-03]
qpos:  [-4.45639944e-03  1.25465391e+00  1.32690946e-03 -2.09982656e-03
 -3.97515743e-03  1.73076348e-03 -1.07423260e-03  1.69846067e-03
 -4.40161031e-03]
"""


env = gym.make("Pendulum-v0")
# print(env.kinematics_integrator)
h = env.observation_space.high
l = env.observation_space.low
action = env.action_space.sample()
env.reset()
print(env.step(action))
print(env.state)
print(h, l)
"""
[4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]
[-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]
"""
# import numpy as np
# a = np.array([ -0.64011951  , 4.33959753 , -2.16438958 , -13.78775058])
#
# print(np.clip(a, l, h))

# print(env.observation_space)
# state = [1,2,3,4]
# ten_s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
# print(torch.is_tensor(ten_s))
# import numpy as np
# a = [ 0.0471, -0.0330, -0.0497,  0.0000,  0.0000,  0.0000]
# b = [a] * 10
# b = np.array(b)
# # print(b.shape[0])
# # c = np.random.choice(b.shape[0], 4, replace=False)
# # print(b[c])
#
# k = np.pad(b,pad_width=((0,0),(1,0)),mode='constant',constant_values=0)
# print(b.shape == b.shape)
# buf_state = [[1,2,3,4],[2,3,4,5]]
# import numpy as np
# print(np.random.sample(np.array(buf_state)))
