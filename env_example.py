import os
import time
import argparse
import numpy as np
import torch

from env import Env, EnvBatcher
from args import get_parser


args = get_parser()

train_envs_name = args.envs.split("~")
base_environment = train_envs_name[0].split("-")[0]
print(args.envs)
print("all train tasks:", train_envs_name)
args.device = torch.device(args.device)

# Initialise training environment and experience replay memory
print("our envs are:", train_envs_name)
task_parameters = []
if args.env_type == 'dmc':
    train_envs = []
    test_envs = []
    for i in range(len(train_envs_name)):
        train_envs.append(Env(train_envs_name[i], args.symbolic_env, args.seed,
                              args.max_episode_length, args.action_repeat, args.bit_depth))

    test_envs_name = []
    if args.test_envs is not None:
        test_envs_name = args.test_envs.split("~")
        for i in range(len(test_envs_name)):
            test_envs.append(Env(test_envs_name[i], args.symbolic_env, args.seed,
                                args.max_episode_length, args.action_repeat, args.bit_depth))
    print("train envs name:", train_envs_name)
    print("test envs name:", test_envs_name)

    train_environment_number = len(train_envs_name)
    args.train_environment_number = train_environment_number
    test_environment_number = len(test_envs_name)
    print("We have {} train environments".format(train_environment_number))
    print("We have {} test environments".format(test_environment_number))
    print("environments are loaded")

    action_size = train_envs[0].action_size
    observation_size = train_envs[0].observation_size
    args.action_size = action_size
    print('action size of all tasks:', action_size)
elif args.env_type == 'mujoco':
    print("our envs are:", train_envs_name)

    env_name = args.envs
    all_envs = Env(env_name, args.symbolic_env, args.seed, args.max_episode_length,
                   args.action_repeat, args.bit_depth)
    all_tasks = all_envs._env.get_all_task_idx()
    train_tasks = []
    test_tasks = []
    if env_name == "cheetah-dir":
        train_tasks = all_tasks
        train_envs_name = ["cheetah-forward", "cheetah-backward"]
        test_tasks = []
        test_envs_name = []
        task_parameters = [1.0, -1.0]
        task_parameters = torch.tensor(task_parameters, dtype=torch.float).to(device=args.device)
    elif env_name == "ant-dir":
        train_tasks = all_tasks
        train_envs_name = ["ant-forward", "ant-backward"]
        test_tasks = []
        test_envs_name = []
        task_parameters = [1.0, -1.0]
        task_parameters = torch.tensor(task_parameters, dtype=torch.float).to(device=args.device)
    elif env_name == "cheetah-vel":
        train_tasks = all_tasks[:100]
        test_tasks = all_tasks[100:]
        train_envs_name = ["cheetah-vel-train-"] * 100
        test_envs_name = ["cheetah-vel-test-"] * 30
        for _ in range(100):
            train_envs_name[_] = train_envs_name[_] + \
                                 str(all_envs._env.tasks[train_tasks[_]]['velocity'])
            task_parameters.append(all_envs._env.tasks[train_tasks[_]]['velocity'])
        for _ in range(30):
            test_envs_name[_] = test_envs_name[_] + \
                                str(all_envs._env.tasks[test_tasks[_]]['velocity'])
        task_parameters = torch.tensor(task_parameters, dtype=torch.float).to(device=args.device)
    elif env_name == "humanoid-dir":
        # obs dim 376, action dim 17
        train_tasks = all_tasks[:100]
        test_tasks = all_tasks[100:]
        train_envs_name = ["humanoid-dir-train-"] * 100
        test_envs_name = ["humanoid-dir-test-"] * 30
        for _ in range(100):
            train_envs_name[_] = train_envs_name[_] + \
                                 str(all_envs._env.tasks[train_tasks[_]]['goal'])
            task_parameters.append(all_envs._env.tasks[train_tasks[_]]['goal'])
        for _ in range(30):
            test_envs_name[_] = test_envs_name[_] + \
                                str(all_envs._env.tasks[test_tasks[_]]['goal'])
        task_parameters = torch.tensor(task_parameters, dtype=torch.float).to(device=args.device)
    elif env_name == "ant-goal":
        train_tasks = all_tasks[:100]
        test_tasks = all_tasks[100:]
        train_envs_name = ["ant-goal-train-"] * 100
        test_envs_name = ["ant-goal-test-"] * 30
        for _ in range(100):
            train_envs_name[_] = train_envs_name[_] + \
                                 str(all_envs._env.tasks[train_tasks[_]]['goal'][0]) + \
                                 str(all_envs._env.tasks[train_tasks[_]]['goal'][1])
        for _ in range(30):
            test_envs_name[_] = test_envs_name[_] + \
                                str(all_envs._env.tasks[test_tasks[_]]['goal'][0]) + \
                                str(all_envs._env.tasks[test_tasks[_]]['goal'][1])
    else:
        raise Exception('Current env is {}, but we only support envs including '
                        'cheetah-dir, ant-dir, cheetah-vel, humanoid-vel, ant-goal'.format(env_name))

    observation_size = int(np.prod(all_envs._env.observation_space.shape))
    action_size = int(np.prod(all_envs._env.action_space.shape))
    args.action_size = action_size
    reward_size = 1
    print('obs dim:', observation_size)
    print('action dim:', action_size)

    if args.not_use_enc:
        if args.embedding_size != observation_size:
            raise Exception('We do not use encoder, thus embedding size must be '
                            'the same as the observation_size, and now embedding_size={}, '
                            'observation_size={}'.format(args.embedding_size, observation_size))
    print("train envs name:", train_envs_name)
    print("test envs name:", test_envs_name)

    train_environment_number = len(train_tasks)
    test_environment_number = len(test_tasks)
    print("We have {} train environments".format(train_environment_number))
    print("We have {} test environments".format(test_environment_number))
    print("environments are loaded")
else:
    raise NotImplementedError

import pdb; pdb.set_trace()
## 用法
# DMControl多task环境
# xvfb-run -a python env_example.py --env-type dmc --envs mywalker-stand~mywalker-walk~mywalker-prostrate~mywalker-flip
# xvfb-run -a python env_example.py --env-type dmc --envs  p4echeetah-run~p4echeetah-run_back~p4echeetah-flip_forward~p4echeetah-flip_backward
# xvfb-run -a python env_example.py --env-type dmc --envs cartpole-balance~cartpole-balance_sparse

# 我们有一个变量叫做train_envs,是一个list of envs
# action_size = train_envs[0].action_size
# observation_size = train_envs[0].observation_size
# observation, done, t = train_envs[current_number].reset(), False, 0
# action = train_envs[current_number].sample_random_action()
# next_observation, reward, done = train_envs[current_number].step(action)

# DMControl任务泛化
# xvfb-run -a python env_example.py --env-type dmc --envs "mypendulum-swingup_angle-n-0.95-n-0.9~mypendulum-swingup_angle-n-0.85-n-0.8~mypendulum-swingup_angle-n-0.8-n-0.75~mypendulum-swingup_angle-n-0.7-n-0.65" --test-envs "mypendulum-swingup_angle-n-0.9-n-0.85~mypendulum-swingup_angle-n-0.75-n-0.7"
# xvfb-run -a python env_example.py --env-type dmc --envs  "mycheetah-run_speed-0.5-0.2~mycheetah-run_speed-1.5-0.2~mycheetah-run_speed-2.0-0.2~mycheetah-run_speed-3.0-0.2" --test-envs "mycheetah-run_speed-1.0-0.2~mycheetah-run_speed-2.5-0.2"
# xvfb-run -a python env_example.py --env-type dmc --envs "mywalker-walk_speed-0.5-0.2~mywalker-walk_speed-1.5-0.2~mywalker-walk_speed-2.0-0.2~mywalker-walk_speed-3.0-0.2" --test-envs "mywalker-walk_speed-1.0-0.2~mywalker-walk_speed-2.5-0.2"

# 我们有两个变量叫做train_envs,test_envs，都是list，用法是一样的
# action_size = train_envs[0].action_size
# observation_size = train_envs[0].observation_size
# observation, done, t = train_envs[current_number].reset(), False, 0
# action = train_envs[current_number].sample_random_action()
# next_observation, reward, done = train_envs[current_number].step(action)


# mujoco环境
# xvfb-run -a python env_example.py --env-type mujoco --envs cheetah-dir 
# xvfb-run -a python env_example.py --env-type mujoco --envs cheetah-vel
# xvfb-run -a python env_example.py --env-type mujoco --envs humanoid-dir

# 我们有一个变量叫做all_envs,用法为
# current_number为任务id，0,1,2,3...
# all_envs._env.reset_task(train_tasks[current_number])
# all_envs._env.reset_task(test_tasks[current_number])
# observation, done, t = all_envs.reset(), False, 0
# action = all_envs.sample_random_action()
# next_observation, reward, done, info = all_envs.step(action)