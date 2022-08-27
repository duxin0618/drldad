import os
import time
import torch
import numpy as np
import multiprocessing as mp

from DaDRL.DaD.learn_model_f import ModelBased
from elegantrl.train.config import build_env
from DaDRL.train.evaluator import Evaluator
from DaDRL.DaD.replay_buffer import ReplayBufferList, ModelTrainBufferList

def train_and_evaluate(args, threshold):
    torch.set_grad_enabled(False)
    args.init_before_training()
    gpu_id = args.learner_gpus


    """init"""
    env = build_env(args.env, args.env_func, args.env_args)

    agent = init_agent(args, gpu_id, env)
    buffer = init_buffer(args, gpu_id)

    evaluator = init_evaluator(args, gpu_id)

    """ DaD"""
    MB = ModelBased(args)
    train_model_buffer = init_model_trainbuffer(int(args.target_step))
    useDaD = args.useDaD
    useTrainModel = args.useTrainModel
    if_save = None
    agent.state = env.reset()

    if args.if_off_policy:
        trajectory = agent.explore_env(env, args.target_step)
        buffer.update_buffer((trajectory,))

    """start training"""
    cwd = args.cwd
    break_step = args.break_step  # 为replaybuffer准备的
    target_step = args.target_step
    if_allow_break = args.if_allow_break
    model_training_argu_rollout_length = args.model_training_argu_rollout_length
    del args

    explore_rewards = list()

    """ record model error"""
    init_dad_model_errors = list()
    min_dad_model_errors = list()
    model_loss = list()
    # reaL_steps, train_model_sample_reward, fake_env_steps, fake_env_reward, eval_reward
    record_compete_info = list()

    threshold = threshold
    if_train = True

    real_explore_steps = 0
    rollout_steps = 0

    # control print dad info
    control_out_num = 1e3

    # train number
    train_number = 0
    while if_train:

        trajectory, model_train_trajectory, explore_reward ,raw_rewards, steps_i = agent.explore_env(env, target_step)
        explore_rewards.append(explore_reward)
        new_exp_rew = explore_reward
        real_explore_steps += steps_i
        explore_reward_mean = np.mean(explore_reward)
        print("eval_reward_mean: ", np.mean(explore_reward))
        print("real_explore_steps is :", real_explore_steps)
        cur_train_min_rew = np.min(new_exp_rew)
        cur_train_max_rew = np.max(new_exp_rew)
        cur_train_mean_rew = np.mean(new_exp_rew)
        rew_info = [cur_train_min_rew, cur_train_max_rew, cur_train_mean_rew]
        print("rew_info", rew_info)
        '''
        Model train
        '''
        useTrainModel = useTrainModel
        if useTrainModel:
            if real_explore_steps < 1000:
                continue
            p = 0; q = min(p + model_training_argu_rollout_length, steps_i)
            cur_items = list(map(list, zip((model_train_trajectory))))
            while p < steps_i:
                train_model_buffer.augment_buffer(cur_items[p: q], is_map=True)
                loss, t = MB.train(train_model_buffer)
                if not t:
                    continue
                model_loss.append(loss)
                print("model_loss : ", loss)
                p = q
                q = min(p + model_training_argu_rollout_length, steps_i)

                if useDaD:
                    min_model_error, init_model_error = MB.get_dad_iter_error()
                    init_dad_model_errors.append(round(init_model_error, 2))
                    min_dad_model_errors.append(round(min_model_error, 2))
                for i in range(int(5)):
                    rollout_trajectory, rollout_step, mean_model_explore_reward = MB.explore_env(env, target_step, train_model_buffer, agent.act, rew_info)
                    print("cur_rollout_steps is: ", rollout_step)
                    if rollout_trajectory is False:
                        continue
                    if rollout_step < 128:
                        continue
                    steps, r_exp = buffer.update_buffer((rollout_trajectory,))
                    train_number += 1
                    rollout_steps += rollout_step
                    torch.set_grad_enabled(True)
                    logging_tuple = agent.update_net(buffer)
                    torch.set_grad_enabled(False)

                    print("model train number is :", train_number,"real_explore_steps", real_explore_steps, " cur_rollout_steps is ", rollout_step, "sum_rollout_steps: ", rollout_steps)

                    (if_reach_goal, if_save, r_avg) = evaluator.evaluate_save_and_plot(
                        agent.act, steps, r_exp, logging_tuple
                    )
                    # reaL_steps, train_model_sample_reward, fake_env_steps, fake_env_reward, eval_reward
                    record_compete_info.append((real_explore_steps, explore_reward_mean, rollout_steps,
                                                mean_model_explore_reward, r_avg))
                    dont_break = not if_allow_break
                    not_reached_goal = not if_reach_goal
                    stop_dir_absent = not os.path.exists(f"{cwd}/stop")
                    if_train = (
                        (dont_break or not_reached_goal)
                        and evaluator.total_step <= break_step
                        and stop_dir_absent
                    )

                    if steps // (control_out_num) > 0 :
                        control_out_num = control_out_num + 1e3
                        plot_info("Model loss", model_loss, "model_loss", cwd, "model_loss")
                        """store train loss"""
                        store_npy_data(("Model loss", model_loss), cwd=cwd, name="model_loss")
                        """store compete info"""
                        np.save(f"{cwd}/record_model_opt_info.npy", record_compete_info)
                        if useDaD:

                            min_train_error, train_error, train_recession_error = MB.get_dad_model_error()

                            """draw min_error and init_error curve"""
                            plot_info("Dad model Init_error and Min_error", init_dad_model_errors, "init_dad_model_errors", cwd,
                                      "dad error",
                                      min_dad_model_errors, "min_dad_model_errors")

                            """store min_error and init_error data"""
                            store_npy_data(("min_dad_error", min_dad_model_errors), ("init_dad_error", init_dad_model_errors), cwd=cwd,
                                           name="dad model error")

                            """store train error"""
                            store_npy_data(("train_error", train_error), cwd=cwd, name="dad_train_error")

                            """store train recession error"""
                            store_npy_data(("train_recession_error", train_recession_error), cwd=cwd, name="train_recession_error")
                    # if mean_model_explore_reward > cur_train_max_rew:
                    #     break
    print(f"| UsedTime: {time.time() - evaluator.start_time:.0f} | SavedDir: {cwd}")
    agent.save_or_load_agent(cwd, if_save=if_save)
    buffer.save_or_load_history(cwd, if_save=True) if agent.if_off_policy else None

    """record add dad steps number"""
    np.save(cwd + "/" + "modeladdstepsnumber", rollout_steps)
    evaluator.save_explorerewards_curve_plot_and_npy(cwd, explore_rewards, useDaD, threshold)

    if useDaD:
        min_train_error, train_error, train_recession_error = MB.get_dad_model_error()

        """draw min_error and init_error curve"""
        plot_info("Dad model Init_error and Min_error", init_dad_model_errors, "init_dad_model_errors", cwd,
                  "dad error",
                  min_dad_model_errors, "min_dad_model_errors")

        """store min_error and init_error data"""
        store_npy_data(("min_error", min_dad_model_errors), ("init_error", init_dad_model_errors), cwd=cwd,
                       name="dad model error")

        """store train error"""
        store_npy_data(("train_error", train_error), cwd=cwd, name="dad_train_error")

        """store train recession error"""
        store_npy_data(("train_recession_error", train_recession_error), cwd=cwd, name="train_recession_error")

def plot_info(title, y1 : list, labely1, cwd, filename, y2 : list=None, labely2=None):
    y1 = np.asarray(y1)
    # delete Abnormal point
    def delete_no_data(data_array) -> np.array:
        if data_array.size == 0:
            return data_array
        mean = np.mean(data_array)
        std = np.std(data_array)
        preprocessed_data_array = [x for x in data_array if (x > mean - std)]
        preprocessed_data_array = [x for x in preprocessed_data_array if (x < mean + std)]
        return preprocessed_data_array

    import matplotlib.pyplot as plt

    plt.cla()
    y1 = y1.ravel()
    y1mean = np.mean(delete_no_data(y1))
    steps = y1.size
    length = range(steps)

    filename = filename
    title = title
    plt.title(title, fontweight="bold")
    plt.ylabel("dad error")
    plt.xlabel("Episode")
    plt.plot(length, y1, marker="^", linestyle="-", color="r", label=labely1+" -mean: "+str(y1mean))
    if y2 is not None:
        y2 = np.asarray(y2)
        y2 = y2.ravel()
        y2mean = np.mean(y2)
        plt.plot(length, y2, marker="s", linestyle="-", color="b", label=labely2+" -mean: "+str(y2mean))

    plt.legend(loc='upper left', bbox_to_anchor=(0.01, 0.98))
    plt.grid()
    plt.savefig(f"{cwd}/{filename}.jpg")
    # plt.show()

def store_npy_data(*data, cwd, name):
    cur = dict()
    for key, da in data:
        cur[key] = np.array(da)
    np.save(cwd+"/"+name, cur)


def read_npy_data(cwd):
    data = np.load(cwd, allow_pickle=True)

    print(data.item())


def init_agent(args, gpu_id: int, env=None):
    agent = args.agent(args.net_dim, args.state_dim, args.action_dim, gpu_id=gpu_id, args=args)
    agent.save_or_load_agent(args.cwd, if_save=False)

    if env is not None:
        '''assign `agent.states` for exploration'''
        if args.env_num == 1:
            states = [env.reset(), ]
            assert isinstance(states[0], np.ndarray)
            assert states[0].shape in {(args.state_dim,), args.state_dim, (args.state_dim+1,), args.state_dim+1}
        else:
            states = env.reset()
            assert isinstance(states, torch.Tensor)
            assert states.shape == (args.env_num, args.state_dim)
        agent.states = states
    return agent


def init_buffer(args, gpu_id):
    buffer = ReplayBufferList()
    return buffer


def init_model_trainbuffer(max_size):
    buffer = ModelTrainBufferList(max_size=max_size)
    return buffer


def init_evaluator(args, gpu_id):
    eval_func = args.eval_env_func if hasattr(args, "eval_env_func") else args.env_func
    eval_args = args.eval_env_args if hasattr(args, "eval_env_args") else args.env_args
    eval_env = build_env(args.env, eval_func, eval_args)
    evaluator = Evaluator(cwd=args.cwd, agent_id=gpu_id, eval_env=eval_env, args=args)
    return evaluator


"""train multiple process"""


def train_and_evaluate_mp(args):
    args.init_before_training()

    process = list()
    mp.set_start_method(
        method="spawn", force=True
    )  # force all the multiprocessing to 'spawn' methods

    evaluator_pipe = PipeEvaluator()
    process.append(mp.Process(target=evaluator_pipe.run, args=(args,)))

    worker_pipe = PipeWorker(args.worker_num)
    process.extend(
        [
            mp.Process(target=worker_pipe.run, args=(args, worker_id))
            for worker_id in range(args.worker_num)
        ]
    )

    learner_pipe = PipeLearner()
    process.append(
        mp.Process(target=learner_pipe.run, args=(args, evaluator_pipe, worker_pipe))
    )

    for p in process:
        p.start()

    process[-1].join()  # waiting for learner
    process_safely_terminate(process)


class PipeWorker:
    def __init__(self, worker_num):
        self.worker_num = worker_num
        self.pipes = [mp.Pipe() for _ in range(worker_num)]
        self.pipe1s = [pipe[1] for pipe in self.pipes]

    def explore(self, agent):
        act_dict = agent.act.state_dict()

        for worker_id in range(self.worker_num):
            self.pipe1s[worker_id].send(act_dict)

        traj_lists = [pipe1.recv() for pipe1 in self.pipe1s]
        return traj_lists

    def run(self, args, worker_id):
        torch.set_grad_enabled(False)
        gpu_id = args.learner_gpus

        """init"""
        env = build_env(args.env, args.env_func, args.env_args)
        agent = init_agent(args, gpu_id, env)

        """loop"""
        target_step = args.target_step
        if args.if_off_policy:
            trajectory = agent.explore_env(env, args.target_step)
            self.pipes[worker_id][0].send(trajectory)
        del args

        while True:
            act_dict = self.pipes[worker_id][0].recv()
            agent.act.load_state_dict(act_dict)
            trajectory = agent.explore_env(env, target_step)
            self.pipes[worker_id][0].send(trajectory)


class PipeLearner:
    def __init__(self):
        pass

    @staticmethod
    def run(args, comm_eva, comm_exp):
        torch.set_grad_enabled(False)
        gpu_id = args.learner_gpus

        """init"""
        agent = init_agent(args, gpu_id)
        buffer = init_buffer(args, gpu_id)

        """loop"""
        if_train = True
        while if_train:
            traj_list = comm_exp.explore(agent)
            steps, r_exp = buffer.update_buffer(traj_list)

            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)

            if_train, if_save = comm_eva.evaluate_and_save_mp(
                agent.act, steps, r_exp, logging_tuple
            )
        agent.save_or_load_agent(args.cwd, if_save=True)
        print(f"| Learner: Save in {args.cwd}")

        if hasattr(buffer, "save_or_load_history"):
            print(f"| LearnerPipe.run: ReplayBuffer saving in {args.cwd}")
            buffer.save_or_load_history(args.cwd, if_save=True)


class PipeEvaluator:
    def __init__(self):
        self.pipe0, self.pipe1 = mp.Pipe()

    def evaluate_and_save_mp(self, act, steps, r_exp, logging_tuple):
        if self.pipe1.poll():  # if_evaluator_idle
            if_train, if_save_agent = self.pipe1.recv()
            act_state_dict = act.state_dict().copy()  # deepcopy(act.state_dict())
        else:
            if_train = True
            if_save_agent = False
            act_state_dict = None

        self.pipe1.send((act_state_dict, steps, r_exp, logging_tuple))
        return if_train, if_save_agent

    def run(self, args):
        torch.set_grad_enabled(False)
        gpu_id = args.learner_gpus

        """init"""
        agent = init_agent(args, gpu_id)
        evaluator = init_evaluator(args, gpu_id)

        """loop"""
        cwd = args.cwd
        act = agent.act
        break_step = args.break_step
        if_allow_break = args.if_allow_break
        del args

        if_save = False
        if_train = True
        if_reach_goal = False
        temp = 0  # todo
        while if_train:
            act_dict, steps, r_exp, logging_tuple = self.pipe0.recv()

            if act_dict:
                act.load_state_dict(act_dict)
                if_reach_goal, if_save = evaluator.evaluate_save_and_plot(
                    act, steps, r_exp, logging_tuple
                )

                temp += 1
                if temp == 4:  # todo
                    temp = 0
                    torch.save(
                        act.state_dict(), f"{cwd}/actor_{evaluator.total_step:09}.pth"
                    )  # todo
            else:
                evaluator.total_step += steps

            if_train = not (
                (if_allow_break and if_reach_goal)
                or evaluator.total_step > break_step
                or os.path.exists(f"{cwd}/stop")
            )
            self.pipe0.send((if_train, if_save))

        print(
            f"| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}"
        )

        while True:  # wait for the forced stop from main process
            self.pipe0.recv()
            self.pipe0.send((False, False))


def process_safely_terminate(process):
    for p in process:
        try:
            p.kill()
        except OSError as e:
            print(e)
