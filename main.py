import os
import time
import argparse
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image

from env import Env, EnvBatcher
from memory import ExperienceReplay
from models import ActorModel, Encoder, ObservationModel, RewardModel, TransitionModel, \
    ValueModel, bottle, TaskModel
from planner import MPCPlanner
from utils import FreezeParameters, imagine_ahead, lambda_return, lineplot, write_video
from evaluation import sample_one_trajectory
from args import get_parser

# we can run planet/dreamer/dreamer-tv/tad for single task by
# - xvfb-run -a python main.py --algo planet --envs "walker-walk" --action-repeat 2 --device cuda:0 --seed 0
# - xvfb-run -a python main.py --algo dreamer --envs "walker-walk" --action-repeat 2 --device cuda:0 --seed 0
# - xvfb-run -a python main.py --algo tad --envs "walker-walk" --action-repeat 2 --device cuda:0 --seed 0
# we can set --id for the name of file
# For tad, we can set the hyperparameter reward-size for resizing the reward by
# --reward-size 10.0
##########
# If you want to run multitask
# you just need to set --envs "walker-walk~walker-stand"
# If you want to run task generalization
# you just need to set --envs "walker-walk~walker-stand" --test-envs "walker-run"
# For multitask, we can separate our replay buffer by
# --sep-replay-buffer True
# For tad, we can set the weight of task loss by
# --task-weight 1.0
##########
# The task we test
# Relevant Tasks
# walker-stand~walker-walk
# cartpole-balance~cartpole-balance_sparse
# acrobot-swingup~acrobot-swingup_sparse
# quadruped-walk~quaduuped-run
# hopper-stand~hopper-hop
# reacher-easy~reacher-hard
# mywalker-stand~mywalker-prostrate
# mywalker-walk~mywalker-prostrate
# p4echeetah-run~p4echeetah-run_back~p4echeetah-flip_forward~p4echeetah-flip_backward

# Task Generalization
# 1.
# --envs "mypendulum-swingup_angle-n-0.95-n-0.9~mypendulum-swingup_angle-n-0.85-n-0.8
# ~mypendulum-swingup_angle-n-0.8-n-0.75~mypendulum-swingup_angle-n-0.7-n-0.65"
# --test-envs "mypendulum-swingup_angle-n-0.9-n-0.85~mypendulum-swingup_angle-n-0.75-n-0.7"
# 2.
# --envs "mycheetah-run_speed-0.5-0.2~mycheetah-run_speed-1.5-0.2
# ~mycheetah-run_speed-2.0-0.2~mycheetah-run_speed-3.0-0.2"
# --test-envs "mycheetah-run_speed-1.0-0.2~mycheetah-run_speed-2.5-0.2"
# 3.
# --envs "myquadruped-walk_speed-0.5-0.2~myquadruped-walk_speed-1.5-0.2
# ~myquadruped-walk_speed-2.0-0.2~myquadruped-walk_speed-3.0-0.2"
# --test-envs "myquadruped-walk_speed-1.0-0.2~myquadruped-walk_speed-2.5-0.2"

# sparse
# xvfb-run -a python main.py --algo dreamer
# --envs "mycheetah-run_speed_s-0.5-0.2-10~mycheetah-run_speed_s-1.5-0.2-10~
# mycheetah-run_speed_s-2.0-0.2-10~mycheetah-run_speed_s-3.0-0.2-10"
# --test-envs "mycheetah-run_speed_s-1.0-0.2-10~mycheetah-run_speed_s-2.5-0.1-10"
# --action-repeat 2 --id dreamer --device cuda:1 --seed 0
# --episodes 2000 --sep-replay-buffer True --reward-size 0.0 --task-weight 0.0

# ctad = tad + combine context to actor/critic

args = get_parser()

if args.cfg is None:
    print("we have not used extra configs.")
else:
    import yaml
    with open('./config/' + args.cfg + '.yaml', 'r') as f:
        args_dict = vars(args)
        cont = f.read()
        args_cfg = yaml.safe_load(cont)
        print("args", args_dict)
        print("file", args_cfg)
        if not args_dict.keys() >= args_cfg.keys():
            raise ValueError("the config file to overwrite the args contains a wrong arg")
        args_dict.update(args_cfg)
        print("updated args", args_dict)
        args = argparse.Namespace(**args_dict)

args.use_paper_reward = False
args.use_encode_r_transition = False
args.task_vector = False
if args.algo == "dreamer":
    print("We use DREAMER")
elif args.algo == 'planet':
    print("We use PLANET")
elif args.algo == "tad" or args.algo == "ctad":
    print("We use TAD")
    args.use_paper_reward = True
    args.use_encode_r_transition = True
elif args.algo == "dreamer-tv":
    print("We use Dreamer with task id")
    args.task_vector = True
    assert (args.test_envs is None), 'We can not set id for test tasks'
else:
    raise Exception('Current algorithm is {}, but we only support algorithm including '
                    'planet, dreamer, tad, ctad, dreamer-tv'.format(args.algo))

if args.id == "":
    args.id = args.algo
if args.cfg is not None:
    args.id = args.id + '_' + args.cfg
print("our experiment id is:", args.id)

# print('All Options')
# for k, v in vars(args).items():
#     print(' ' * 10 + k + ': ' + str(v))

train_envs_name = args.envs.split("~")
base_environment = train_envs_name[0].split("-")[0]
print(args.envs)
print("all train tasks:", train_envs_name)

# Setup file name
if args.sep_replay_buffer:
    alg_path = args.id + '_sep_' + str(args.reward_size) + '_' + str(args.task_weight)
else:
    alg_path = args.id + '_' + str(args.reward_size) + '_' + str(args.task_weight)
if args.test_envs is None:
    env_path = args.envs.replace("~", "_")
else:
    env_path = args.envs.replace("~", "_") + "---" + args.test_envs.replace("~", "_")

alg_path = alg_path + '_' + args.task_loss_type + '_' + str(args.tau) + '_' + str(args.context_dim)

results_dir = os.path.join('results', env_path, alg_path, str(args.seed))
print('results dir is', results_dir)
os.makedirs(results_dir, exist_ok=True)

# setup random seed and cuda/cpu
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() and args.device != 'cpu':
    print("using CUDA")
    torch.cuda.manual_seed(args.seed)
else:
    print("using CPU")
args.device = torch.device(args.device)

# setup metrics
metrics = {
    'steps': [],
    'episodes': [],
    'observation_loss': [],
    'reward_loss': [],
    'kl_loss': [],
    'actor_loss': [],
    'value_loss': [],
    'task_loss': [],
    'train_rewards': [],
    'evaluate_episodes': [],
    'test_rewards': [],
    'evaluate_train_envs_average_rewards': [],
    'evaluate_test_envs_average_rewards': [],
    'evaluate_train_envs_all_rewards': [],
    'evaluate_test_envs_all_rewards': [],
    'evaluate_steps': [],
    'all_test_rewards': [],
}

summary_name = results_dir + "/{}_{}_log"
writer = SummaryWriter(summary_name.format(base_environment, args.id))
print("writer is ready")

# Initialise training environment and experience replay memory
print("our envs are:", train_envs_name)
task_parameters = []
if args.env_type == 'dmc':
    train_envs = []
    for i in range(len(train_envs_name)):
        train_envs.append(Env(train_envs_name[i], args.symbolic_env, args.seed,
                              args.max_episode_length, args.action_repeat, args.bit_depth))

    test_envs_name = []
    if args.test_envs is not None:
        test_envs_name = args.test_envs.split("~")
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

if args.sep_replay_buffer:
    print('we use separate replay buffer')
    print('original batch size:', args.batch_size)
    args.batch_size = int((args.batch_size + train_environment_number - 1) /
                          train_environment_number) * train_environment_number
    print('current batch size:', args.batch_size)

D = None
Ds = None
assert args.seed_episodes % args.num_tasks_sample == 0
assert args.test_interval % args.num_tasks_sample == 0
assert args.checkpoint_interval % args.num_tasks_sample == 0
if args.experience_replay != '' and os.path.exists(args.experience_replay):
    print("We use the stored replay buffer")
    D = torch.load(args.experience_replay)
    metrics['steps'], metrics['episodes'] = [D.steps] * D.episodes, list(range(1, D.episodes + 1))
else:
    if args.sep_replay_buffer:
        Ds = []
        for _ in range(train_environment_number):
            Ds.append(ExperienceReplay(args.experience_size, args.symbolic_env,
                                       observation_size, action_size,
                                       args.bit_depth, args.device))
    else:
        D = ExperienceReplay(
            args.experience_size, args.symbolic_env, observation_size,
            action_size, args.bit_depth, args.device
        )

    # Initialise dataset D with S random seed episodes
    for s in range(1, args.seed_episodes + 1):
        current_number = (s - 1) % train_environment_number
        print("#" * 10, "This is episode {}, current is the environment:".format(s),
              train_envs_name[current_number], "#" * 10)
        if args.env_type == 'dmc':
            observation, done, t = train_envs[current_number].reset(), False, 0
        elif args.env_type == 'mujoco':
            all_envs._env.reset_task(current_number)
            observation, done, t = all_envs.reset(), False, 0
        else:
            raise NotImplementedError

        while not done:
            if args.env_type == 'dmc':
                action = train_envs[current_number].sample_random_action()
                next_observation, reward, done = train_envs[current_number].step(action)
            elif args.env_type == 'mujoco':
                action = all_envs.sample_random_action()
                next_observation, reward, done, info = all_envs.step(action)
            else:
                raise NotImplementedError

            if args.sep_replay_buffer:
                Ds[current_number].append(observation, action, reward, done)
            else:
                D.append(observation, action, reward, done)
            observation = next_observation
            t += 1
        metrics['steps'].append(t * args.action_repeat +
                                (0 if len(metrics['steps']) == 0
                                 else metrics['steps'][-1]))
        metrics['episodes'].append(s)
print("experience replay buffer is ready")

# Initialise model parameters randomly
if not args.task_vector:
    embedding_size = args.embedding_size
else:
    embedding_size = args.embedding_size + train_environment_number
transition_model = TransitionModel(args.belief_size, args.state_size,
                                   action_size, args.hidden_size, embedding_size,
                                   args.dense_activation_function,
                                   encoder_r=args.use_encode_r_transition).to(device=args.device)
if args.context_dim == 0:
    args.context_dim = train_environment_number
if args.task_loss_type == 'continuous_mse':
    args.context_dim = 1
task_model = TaskModel(args.belief_size, args.state_size, args.hidden_size,
                       args.dense_activation_function,
                       task_number=args.context_dim).to(device=args.device)
observation_model = ObservationModel(args.symbolic_env, observation_size,
                                     args.belief_size, args.state_size, args.embedding_size,
                                     args.cnn_activation_function, ).to(device=args.device)
reward_model = RewardModel(args.belief_size, args.state_size, args.hidden_size,
                           args.dense_activation_function).to(device=args.device)
encoder = Encoder(args.symbolic_env, observation_size, args.embedding_size,
                  args.cnn_activation_function).to(device=args.device)

param_list = (list(transition_model.parameters()) + list(observation_model.parameters())
              + list(reward_model.parameters()) + list(encoder.parameters())
              + list(task_model.parameters()))

if args.algo == "ctad":
    actor_model = ActorModel(args.belief_size+args.context_dim, args.state_size,
                             args.hidden_size,
                             action_size, args.dense_activation_function).to(device=args.device)
    value_model = ValueModel(args.belief_size+args.context_dim, args.state_size,
                             args.hidden_size,
                             args.dense_activation_function).to(device=args.device)
else:
    actor_model = ActorModel(args.belief_size, args.state_size,
                             args.hidden_size,
                             action_size, args.dense_activation_function).to(device=args.device)
    value_model = ValueModel(args.belief_size, args.state_size,
                             args.hidden_size,
                             args.dense_activation_function).to(device=args.device)

value_actor_param_list = list(value_model.parameters()) + list(actor_model.parameters())
params_list = param_list + value_actor_param_list

print("transition, observation, reward, encoder, actor, value models are ready")

model_optimizer = optim.Adam(param_list,
                             lr=0 if args.learning_rate_schedule != 0
                             else args.model_learning_rate,
                             eps=args.adam_epsilon)
actor_optimizer = optim.Adam(actor_model.parameters(),
                             lr=0 if args.learning_rate_schedule != 0
                             else args.actor_learning_rate,
                             eps=args.adam_epsilon, )
value_optimizer = optim.Adam(value_model.parameters(),
                             lr=0 if args.learning_rate_schedule != 0
                             else args.value_learning_rate,
                             eps=args.adam_epsilon, )
if args.models != '' and os.path.exists(args.models):
    print("loading pre-trained models")
    model_dicts = torch.load(args.models)
    transition_model.load_state_dict(model_dicts['transition_model'])
    observation_model.load_state_dict(model_dicts['observation_model'])
    reward_model.load_state_dict(model_dicts['reward_model'])
    encoder.load_state_dict(model_dicts['encoder'])
    actor_model.load_state_dict(model_dicts['actor_model'])
    value_model.load_state_dict(model_dicts['value_model'])
    task_model.load_state_dict(model_dicts['task_model'])
    model_optimizer.load_state_dict(model_dicts['model_optimizer'])

print('our algorithm is', args.algo)
if args.algo == "dreamer" or args.algo == "dreamer-tv" or \
        args.algo == "tad" or args.algo == 'ctad':
    planner = actor_model
elif args.algo == 'planet':
    planner = MPCPlanner(
        action_size,
        args.planning_horizon,
        args.optimisation_iters,
        args.candidates,
        args.top_candidates,
        transition_model,
        reward_model,
    )
else:
    raise Exception('Current algorithm is {}, but we only support algorithm including '
                    'planet, dreamer, tad, dreamer-tv'.format(args.algo))

# Global prior N(0, I)
global_prior = Normal(torch.zeros(args.batch_size, args.state_size, device=args.device),
                      torch.ones(args.batch_size, args.state_size, device=args.device), )
# Allowed deviation in KL divergence
free_nats = torch.full((1,), args.free_nats, device=args.device)
print("models and planners are ready")

# Training (and testing)
episode = metrics['episodes'][-1]
BEST_RETURN = -1000000.0
BEST_INDEX = -1
BEST_TRAIN_RETURN = -1000000.0
BEST_TEST_RETURN = -1000000.0
model_modules = (transition_model.modules + encoder.modules + task_model.modules +
                 observation_model.modules + reward_model.modules)
current_number = episode % train_environment_number
while episode < args.episodes:
    start_time = time.time()
    print("#" * 10, "This is episode {}, timesteps: {}".
          format(episode + 1, metrics['steps'][-1]), "#" * 10)

    # Model fitting
    print("current is training loop")
    losses = []
    for s in range(args.collect_interval):
        # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly
        # at random from the dataset (including terminal flags)
        # batch size: default 50
        # chunk size: default is 50, we set it as 64 to see the shape
        task_index = []
        if args.sep_replay_buffer:
            all_observations = []
            all_actions = []
            all_rewards = []
            all_nonterminals = []
            for _ in range(train_environment_number):
                current_observations, current_actions, current_rewards, current_nonterminals = \
                    Ds[_].sample(int(args.batch_size / train_environment_number), args.chunk_size)
                all_observations.append(current_observations)
                all_actions.append(current_actions)
                all_rewards.append(current_rewards)
                all_nonterminals.append(current_nonterminals)
                task_index = task_index + [_] * int(args.batch_size / train_environment_number)
            observations = torch.cat(all_observations, dim=1)
            actions = torch.cat(all_actions, dim=1)
            rewards = torch.cat(all_rewards, dim=1)
            nonterminals = torch.cat(all_nonterminals, dim=1)

            # chunk_size * batch_size: 64 * 50, trianing task num: 2
            # print(all_observations[0].shape  # [64, 25, 3, 64, 64]
            # print(all_actions[0].shape)  # [64, 25, 6]
            # print(all_rewards[0].shape)  # [64, 25]
            # print(all_nonterminals[0].shape)  # [64, 25, 1]
            # print(observations.shape) # [64, 50, 3, 64, 64]
            # print(actions.shape) # [64, 50, 6]
            # print(rewards.shape) # [64, 50]
            # print(nonterminals.shape) # [64, 50, 1]
        else:
            observations, actions, rewards, nonterminals = D.sample(
                args.batch_size, args.chunk_size
            )  # Transitions start at time t = 0

            # chunk_size * batch_size: 64 * 50
            # print('o', observations.shape) # [64, 50, 3, 64, 64]
            # print('a', actions.shape) # [64, 50, 6]
            # print('r', rewards.shape) # [64, 50]
            # print('n', nonterminals.shape) # [64, 50, 1]

        # Create initial belief and state for time t = 0
        init_belief = torch.zeros(args.batch_size, args.belief_size, device=args.device)
        init_state = torch.zeros(args.batch_size, args.state_size, device=args.device)

        if args.env_type == 'dmc':
            # print(bottle(encoder, (observations[1:],)).shape) # 49, 50, 1024
            encoded_observations = bottle(encoder, (observations[1:],))
        elif args.env_type == 'mujoco':
            if not args.not_use_enc:
                encoded_observations = bottle(encoder, (observations[1:],))
            else:
                encoded_observations = observations[1:]
        else:
            raise NotImplementedError

        # print('task', task_index)
        if args.task_vector:
            task_index_vector = torch.zeros(args.chunk_size - 1, args.batch_size,
                                            train_environment_number,
                                            device=args.device)
            for _ in range(len(task_index)):
                task_index_vector[:, _, task_index[_]] = 1.0
            encoded_observations = torch.cat((encoded_observations,
                                              task_index_vector), dim=2)

        # Update belief/state using posterior from previous belief/state,
        # previous action and current observation (over entire sequence at once)
        if not args.use_encode_r_transition:
            current_rewards = None
        else:
            current_rewards = args.reward_size * \
                              rewards.reshape(rewards.shape[0], rewards.shape[1], 1)
            current_rewards = current_rewards[:-1]

        (beliefs, prior_states, prior_means, prior_std_devs,
         posterior_states, posterior_means, posterior_std_devs,) = \
            transition_model(init_state, actions[:-1], init_belief,
                             encoded_observations, nonterminals[:-1],
                             current_rewards)

        # Calculate observation likelihood, reward likelihood and KL losses
        # (for t = 0 only for latent overshooting);
        # sum over final dims, average over batch and time
        # (original implementation, though paper seems to miss 1/T scaling?)

        # calculate observation likelihood
        # worldmodel_LogProbLoss default: false
        if args.worldmodel_LogProbLoss:
            observation_dist = Normal(bottle(observation_model, (beliefs, posterior_states)), 1)
            observation_loss = (
                -observation_dist.log_prob(observations[1:])
                .sum(dim=2 if args.symbolic_env else (2, 3, 4))
                .mean(dim=(0, 1))
            )
        else:
            observation_loss = (
                F.mse_loss(bottle(observation_model, (beliefs, posterior_states)),
                           observations[1:], reduction='none')
                .sum(dim=2 if args.symbolic_env else (2, 3, 4))
                .mean(dim=(0, 1))
            )

        # calculate reward likelihood
        if args.use_paper_reward:
            if args.worldmodel_LogProbLoss:
                reward_dist = Normal(bottle(reward_model, (beliefs, posterior_states)), 1)
                reward_loss = -reward_dist.log_prob(rewards[1:]).mean(dim=(0, 1))
            else:
                reward_loss = F.mse_loss(bottle(reward_model, (beliefs, posterior_states)),
                                         rewards[1:], reduction='none').mean(dim=(0, 1))
        else:
            if args.worldmodel_LogProbLoss:
                reward_dist = Normal(bottle(reward_model, (beliefs, posterior_states)), 1)
                reward_loss = -reward_dist.log_prob(rewards[:-1]).mean(dim=(0, 1))
            else:
                reward_loss = F.mse_loss(bottle(reward_model, (beliefs, posterior_states)),
                                         rewards[:-1], reduction='none').mean(dim=(0, 1))

        # calculate transition loss
        div = kl_divergence(Normal(posterior_means, posterior_std_devs),
                            Normal(prior_means, prior_std_devs)).sum(dim=2)
        # Note that normalisation by overshooting distance and
        # weighting by overshooting distance cancel out
        kl_loss = torch.max(div, free_nats).mean(dim=(0, 1))

        # calculate task loss
        task_loss = None
        # args.task_loss_type = 'cross_entropy'
        # args.tau = 0.1

        if args.task_weight == 0.0:
            pass
        elif args.task_loss_type == 'cross_entropy':
            assert args.sep_replay_buffer
            # assert (args.task_weight > 0.0), \
            #     'args.task_weight must > 0.0'
            # print("weight:", args.task_weight)
            task_hidden = bottle(task_model, (beliefs, posterior_states))
            task_index = torch.tensor(task_index, dtype=torch.int64).to(device=args.device)
            task_loss = 0
            # print(task_hidden.shape) # [49, 52, 4]
            # print(task_index.shape) # [52]
            for chunk_index in range(task_hidden.shape[0]):
                # F.cross_entropy == -log_softmax.mean()
                # maximize log_softmax == minimize cross_entropy
                task_loss += F.cross_entropy(task_hidden[chunk_index], task_index)
            task_loss = task_loss / task_hidden.shape[0]
        elif args.task_loss_type == 'continuous_mse':
            assert args.sep_replay_buffer
            task_hidden = bottle(task_model, (beliefs, posterior_states))
            task_index = task_parameters[task_index]
            
            task_loss = nn.MSELoss()(task_hidden[:, :, 0], task_index.view(1, task_index.shape[0]).expand_as(task_hidden[:, :, 0]))
            task_loss = task_loss
        elif args.task_loss_type == 'supervised_contrastive':
            assert (args.tau > 0.0), \
                'args.tau > 0.0'
            # posterior_states.shape # chunk_size-1, batch_size, hidden_size
            # task_index.shape # batch_size
            # print(task_index)

            task_hidden = bottle(task_model, (beliefs, posterior_states))

            temperature = args.tau
            # query: (b,c), task_index: (b, 1)
            # label positives samples
            # labels = torch.argmax(contexts, dim=-1)  # (b, 1)
            labels = torch.tensor(task_index)
            labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).long()  # (b,b)
            labels = labels.to(args.device)

            # discard the main diagonal from both: labels and similarities matrix
            mask = torch.eye(labels.shape[0], dtype=torch.bool).to(args.device)
            labels = labels[~mask].view(labels.shape[0], -1)  # (b,b-1)

            task_loss = 0.0
            # for chunk_index in range(posterior_states.shape[0]):
            #     features = F.normalize(posterior_states[chunk_index], dim=1)
            for chunk_index in range(task_hidden.shape[0]):
                features = F.normalize(task_hidden[chunk_index], dim=1)
                similarity_matrix = torch.matmul(features, features.T)  # (b,b)
                similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)  # (b,b-1)

                similarity_matrix = similarity_matrix / temperature
                similarity_matrix -= torch.max(similarity_matrix, 1)[0][:, None]
                similarity_matrix = torch.exp(similarity_matrix)

                pick_one_positive_sample_idx = torch.argmax(labels, dim=-1, keepdim=True)
                pick_one_positive_sample_idx = torch.zeros_like(labels).scatter_(-1, pick_one_positive_sample_idx, 1)

                positives = torch.sum(similarity_matrix * pick_one_positive_sample_idx, dim=-1, keepdim=True)  # (b,1)
                negatives = torch.sum(similarity_matrix, dim=-1, keepdim=True)  # (b,1)
                eps = torch.as_tensor(1e-6)
                task_loss += torch.mean(-torch.log(positives / (negatives + eps) + eps))  # (b,1)
            # task_loss = task_loss / posterior_states.shape[0]
            task_loss = task_loss / task_hidden.shape[0]
        else:
            raise NotImplementedError

        # default: 0
        if args.global_kl_beta != 0:
            kl_loss += args.global_kl_beta * kl_divergence(
                Normal(posterior_means, posterior_std_devs), global_prior
            ).sum(dim=2).mean(dim=(0, 1))

        # default: 0
        if args.learning_rate_schedule != 0:
            for group in model_optimizer.param_groups:
                group['lr'] = min(group['lr'] + args.model_learning_rate /
                                  args.model_learning_rate_schedule,
                                  args.model_learning_rate)

        # Update model parameters
        model_loss = observation_loss + reward_loss + kl_loss

        if args.sep_replay_buffer and args.task_weight > 0.0:
            # print("we use task loss")
            model_loss += task_loss * args.task_weight

        model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(param_list, args.grad_clip_norm, norm_type=2)
        model_optimizer.step()

        if args.algo == "dreamer" or args.algo == "dreamer-tv" or \
                args.algo == "tad" or args.algo == 'ctad':
            # Dreamer implementation: actor loss calculation and optimization
            with torch.no_grad():
                actor_states = posterior_states.detach()
                actor_beliefs = beliefs.detach()
            with FreezeParameters(model_modules):
                if args.algo == "ctad":
                    # task_contexts = bottle(task_model, (actor_beliefs, actor_states))
                    # actor_beliefs = torch.cat([actor_beliefs, task_contexts], dim=-1)
                    imagination_traj = imagine_ahead(actor_states, actor_beliefs, actor_model,
                                                     transition_model, args.planning_horizon,
                                                     task_model=task_model)
                else:
                    imagination_traj = imagine_ahead(actor_states, actor_beliefs, actor_model,
                                                     transition_model, args.planning_horizon)
            imged_beliefs, imged_prior_states, imged_prior_means, imged_prior_std_devs \
                = imagination_traj
            with FreezeParameters(model_modules + value_model.modules):
                imged_reward = bottle(reward_model, (imged_beliefs, imged_prior_states))
                if args.algo == "ctad":
                    task_contexts = bottle(task_model, (imged_beliefs, imged_prior_states))
                    imged_prior_states = torch.cat([imged_prior_states, task_contexts], dim=-1)
                value_pred = bottle(value_model, (imged_beliefs, imged_prior_states))
            returns = lambda_return(imged_reward, value_pred, bootstrap=value_pred[-1],
                                    discount=args.discount, lambda_=args.disclam)
            actor_loss = -torch.mean(returns)
            # Update model parameters
            actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor_model.parameters(), args.grad_clip_norm, norm_type=2)
            actor_optimizer.step()

            # Dreamer implementation: value loss calculation and optimization
            with torch.no_grad():
                value_beliefs = imged_beliefs.detach()
                value_prior_states = imged_prior_states.detach()
                target_return = returns.detach()
                # Note: here imged_prior_states has concated the task context
                # if args.algo == "ctad":
                #     task_contexts = bottle(task_model, (value_beliefs, value_prior_states))
                #     value_prior_states = torch.cat([value_prior_states, task_contexts], dim=-1)
            # detach the input tensor from the transition network.
            value_dist = Normal(bottle(value_model, (value_beliefs, value_prior_states)), 1)
            value_loss = -value_dist.log_prob(target_return).mean(dim=(0, 1))
            # Update model parameters
            value_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(value_model.parameters(), args.grad_clip_norm, norm_type=2)
            value_optimizer.step()

            # # Store (0) observation loss (1) reward loss (2) KL loss
            # (3) actor loss (4) value loss
            if args.sep_replay_buffer and args.task_weight > 0.0:
                losses.append([observation_loss.item(), reward_loss.item(), kl_loss.item(),
                               actor_loss.item(), value_loss.item(), task_loss.item()])
            else:
                losses.append([observation_loss.item(), reward_loss.item(), kl_loss.item(),
                               actor_loss.item(), value_loss.item()])
        elif args.algo == "planet":
            # # Store (0) observation loss (1) reward loss (2) KL loss
            losses.append([observation_loss.item(), reward_loss.item(), kl_loss.item()])
        else:
            raise Exception('Current algorithm is {}, but we only support algorithm including '
                            'planet, dreamer, tad, ctad, dreamer-tv'.format(args.algo))

    # Update and plot loss metrics
    losses = tuple(zip(*losses))  # similar to reshape
    metrics['observation_loss'].append(losses[0])
    metrics['reward_loss'].append(losses[1])
    metrics['kl_loss'].append(losses[2])
    lineplot(metrics['episodes'][-len(metrics['observation_loss']):],
             metrics['observation_loss'], 'observation_loss', results_dir, )
    lineplot(metrics['episodes'][-len(metrics['reward_loss']):],
             metrics['reward_loss'], 'reward_loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['kl_loss']):],
             metrics['kl_loss'], 'kl_loss', results_dir)
    writer.add_scalar("observation_loss", metrics['observation_loss'][-1][-1], metrics['steps'][-1])
    writer.add_scalar("reward_loss", metrics['reward_loss'][-1][-1], metrics['steps'][-1])
    writer.add_scalar("kl_loss", metrics['kl_loss'][-1][-1], metrics['steps'][-1])
    if args.algo == "dreamer" or args.algo == "dreamer-tv" \
            or args.algo == "tad" or args.algo == "ctad":
        metrics['actor_loss'].append(losses[3])
        metrics['value_loss'].append(losses[4])
        lineplot(metrics['episodes'][-len(metrics['actor_loss']):],
                 metrics['actor_loss'], 'actor_loss', results_dir)
        lineplot(metrics['episodes'][-len(metrics['value_loss']):],
                 metrics['value_loss'], 'value_loss', results_dir)
        writer.add_scalar("actor_loss", metrics['actor_loss'][-1][-1], metrics['steps'][-1])
        writer.add_scalar("value_loss", metrics['value_loss'][-1][-1], metrics['steps'][-1])
        if args.sep_replay_buffer and args.task_weight > 0.0:
            metrics['task_loss'].append(losses[5])
            lineplot(metrics['episodes'][-len(metrics['task_loss']):],
                     metrics['task_loss'], 'task_loss', results_dir)
            writer.add_scalar("task_loss", metrics['task_loss'][-1][-1], metrics['steps'][-1])

    # Data collection
    print("current is data collecting with algorithm {}".format(args.algo))
    for task_sampled_id in range(args.num_tasks_sample):
        current_number = episode % train_environment_number
        with torch.no_grad():
            if args.task_vector:
                task_id = current_number
            else:
                task_id = None

            if args.env_type == 'dmc':
                if args.algo == "ctad":
                    t, total_reward, video_frames, all_data = \
                        sample_one_trajectory(args, train_envs[current_number], planner,
                                              transition_model, encoder, explore_=True,
                                              task_id_=task_id, test=False,
                                              task_model_=task_model)
                else:
                    t, total_reward, video_frames, all_data = \
                        sample_one_trajectory(args, train_envs[current_number], planner,
                                              transition_model, encoder, explore_=True,
                                              task_id_=task_id, test=False)
            elif args.env_type == 'mujoco':
                all_envs._env.reset_task(train_tasks[current_number])
                if args.algo == "ctad":
                    t, total_reward, video_frames, all_data = \
                        sample_one_trajectory(args, all_envs, planner,
                                              transition_model, encoder, explore_=True,
                                              task_id_=task_id, test=False,
                                              task_model_=task_model,
                                              env_type='mujoco',
                                              env_name=env_name)
                else:
                    t, total_reward, video_frames, all_data = \
                        sample_one_trajectory(args, all_envs, planner,
                                              transition_model, encoder, explore_=True,
                                              task_id_=task_id, test=False,
                                              env_type='mujoco',
                                              env_name=env_name)
            else:
                raise NotImplementedError

            for single_data in all_data:
                observation, action, reward, done = single_data
                if args.sep_replay_buffer:
                    # print(observation)
                    Ds[current_number].append(observation, action, reward, done)
                else:
                    D.append(observation, action, reward, done)
            episode += 1
            metrics['steps'].append(t + metrics['steps'][-1])
        if args.sep_replay_buffer:
            print("Episode {}, we store data to Replay Buffer {} in task {}, with reward {}"
                  .format(episode, current_number, train_envs_name[current_number], total_reward))

    # Update and plot train reward metrics
    metrics['episodes'].append(episode)
    metrics['train_rewards'].append(total_reward)
    lineplot(metrics['episodes'][-len(metrics['train_rewards']):],
             metrics['train_rewards'], 'train_rewards', results_dir)
    writer.add_scalar("train_reward", metrics['train_rewards'][-1], metrics['steps'][-1])
    writer.add_scalar("train/episode_reward", metrics['train_rewards'][-1],
                      metrics['steps'][-1] * args.action_repeat)
    print("Episodes: {}, Total_steps: {}, time cost: {}.".
          format(metrics['episodes'][-1], metrics['steps'][-1],
                 time.time() - start_time))

    # Test model
    if episode % args.test_interval == 0:
        start_time = time.time()
        print("#### current is model testing, our algo is", args.algo, "####")

        transition_model.eval()
        observation_model.eval()
        reward_model.eval()
        encoder.eval()
        actor_model.eval()
        value_model.eval()

        if args.env_type == 'mujoco':
            args.test_episodes = 1

        # Initialise parallel test environments
        all_evaluate_train_rewards = []
        average_evaluate_train_rewards = np.zeros((args.test_episodes,))
        all_evaluate_test_rewards = []
        average_evaluate_test_rewards = np.zeros((args.test_episodes,))

        for i in range(train_environment_number+test_environment_number):
            if args.env_type == 'dmc':
                if i < len(train_envs_name):
                    current_task_name = train_envs_name[i]
                    evaluate_envs = EnvBatcher(
                        Env, (current_task_name, args.symbolic_env, args.seed,
                              args.max_episode_length, args.action_repeat, args.bit_depth),
                        {}, args.test_episodes, )
                    print("#"*5, "current environment is training task:", current_task_name, "#"*5)
                else:
                    current_task_name = test_envs_name[i-len(train_envs_name)]
                    evaluate_envs = EnvBatcher(
                        Env, (current_task_name, args.symbolic_env, args.seed,
                              args.max_episode_length, args.action_repeat, args.bit_depth),
                        {}, args.test_episodes, )
                    print("#"*5, "current environment is testing task:", current_task_name, "#"*5)

                with torch.no_grad():
                    if args.task_vector:
                        task_id = i
                    else:
                        task_id = None

                    if args.algo == "ctad":
                        t, total_reward, video_frames, all_data = \
                            sample_one_trajectory(args, evaluate_envs, planner,
                                                  transition_model, encoder, explore_=True,
                                                  task_id_=task_id, test=True,
                                                  observation_model_=observation_model,
                                                  task_model_=task_model)
                    else:
                        t, total_reward, video_frames, all_data = \
                            sample_one_trajectory(args, evaluate_envs, planner,
                                                  transition_model, encoder, explore_=True,
                                                  task_id_=task_id, test=True,
                                                  observation_model_=observation_model)

                    print("test return is:", total_reward)
                    print("mean of return is:", np.mean(total_reward),
                          "std of return is:", np.std(total_reward))
                    if i < len(train_envs_name):
                        average_evaluate_train_rewards += total_reward
                        all_evaluate_train_rewards.append(total_reward.tolist())
                    else:
                        average_evaluate_test_rewards += total_reward
                        all_evaluate_test_rewards.append(total_reward.tolist())
                # write video if applicable
                if not args.symbolic_env and args.save_video:
                    episode_str = str(episode).zfill(len(str(args.episodes)))
                    write_video(video_frames,
                                current_task_name + '_test_episode_%s' % episode_str,
                                results_dir)  # Lossy compression
                    save_image(torch.as_tensor(video_frames[-1]),
                               os.path.join(results_dir, current_task_name +
                                            '_test_episode_%s.png' % episode_str))

                # Close test environments
                evaluate_envs.close()
            elif args.env_type == 'mujoco':
                if i < train_environment_number:
                    all_envs._env.reset_task(train_tasks[i])
                else:
                    all_envs._env.reset_task(test_tasks[i-train_environment_number])

                with torch.no_grad():
                    if args.task_vector:
                        task_id = i
                    else:
                        task_id = None

                    if args.algo == "ctad":
                        t, total_reward, video_frames, all_data = \
                            sample_one_trajectory(args, all_envs, planner,
                                                  transition_model, encoder, explore_=True,
                                                  task_id_=task_id, test=True,
                                                  observation_model_=observation_model,
                                                  task_model_=task_model,
                                                  env_type='mujoco',
                                                  env_name=env_name)
                    else:
                        t, total_reward, video_frames, all_data = \
                            sample_one_trajectory(args, all_envs, planner,
                                                  transition_model, encoder, explore_=True,
                                                  task_id_=task_id, test=True,
                                                  observation_model_=observation_model,
                                                  env_type='mujoco',
                                                  env_name=env_name)
                    print("test return is:", total_reward)
                    print("mean of return is:", np.mean(total_reward),
                          "std of return is:", np.std(total_reward))
                    if i < len(train_envs_name):
                        average_evaluate_train_rewards += total_reward
                        all_evaluate_train_rewards.append(total_reward.tolist())
                    else:
                        average_evaluate_test_rewards += total_reward
                        all_evaluate_test_rewards.append(total_reward.tolist())
            else:
                raise NotImplementedError

        average_evaluate_train_rewards = average_evaluate_train_rewards / train_environment_number
        print("Average return of all training envs:", np.mean(average_evaluate_train_rewards))
        if len(test_envs_name) > 0:
            average_evaluate_test_rewards = average_evaluate_test_rewards / test_environment_number
            print("Average return of all test envs:", np.mean(average_evaluate_test_rewards))

        total_evaluate_rewards = (average_evaluate_train_rewards * train_environment_number +
                                  average_evaluate_test_rewards * test_environment_number)
        average_evaluate_rewards = (total_evaluate_rewards /
                                    (train_environment_number + test_environment_number))

        if np.mean(average_evaluate_rewards) >= BEST_RETURN:
            BEST_RETURN = np.mean(average_evaluate_rewards)
            BEST_INDEX = episode
            BEST_TRAIN_RETURN = np.mean(average_evaluate_train_rewards)
            BEST_TEST_RETURN = np.mean(average_evaluate_test_rewards)
            print('#'*2, 'current we save the best model, the best return is', BEST_RETURN, '#'*2)
            torch.save(
                {
                    'transition_model': transition_model.state_dict(),
                    'observation_model': observation_model.state_dict(),
                    'reward_model': reward_model.state_dict(),
                    'encoder': encoder.state_dict(),
                    'actor_model': actor_model.state_dict(),
                    'value_model': value_model.state_dict(),
                    'task_model': task_model.state_dict(),
                    'model_optimizer': model_optimizer.state_dict(),
                    'actor_optimizer': actor_optimizer.state_dict(),
                    'value_optimizer': value_optimizer.state_dict(),
                },
                os.path.join(results_dir, 'best_model.pth'),
            )
        else:
            print('#'*2, 'the best model is in episode {}, '
                  'and the best return is {}'.format(BEST_INDEX, BEST_RETURN), '#'*2)
            print('the best train return is {}'.format(BEST_TRAIN_RETURN))
            if len(test_envs_name) > 0:
                print('the best test return is {}'.format(BEST_TEST_RETURN))

        # Update and plot reward metrics and save metrics
        metrics['evaluate_episodes'].append(episode)
        metrics['evaluate_train_envs_all_rewards'].append(all_evaluate_train_rewards)
        metrics['evaluate_train_envs_average_rewards'].append(average_evaluate_train_rewards.tolist())
        lineplot(metrics['evaluate_episodes'], metrics['evaluate_train_envs_average_rewards'],
                 'evaluate_train_rewards', results_dir)
        lineplot(np.asarray(metrics['steps'])[np.asarray(metrics['evaluate_episodes']) - 1],
                 metrics['evaluate_train_envs_average_rewards'], 'evaluate_train_rewards_steps',
                 results_dir, xaxis='step')
        writer.add_scalar('evaluate_train_envs_all_rewards',
                          np.mean(average_evaluate_train_rewards),
                          metrics['steps'][-1])
        if len(test_envs_name) > 0:
            metrics['evaluate_test_envs_all_rewards'].append(all_evaluate_test_rewards)
            metrics['evaluate_test_envs_average_rewards'].append(average_evaluate_test_rewards.tolist())
            lineplot(metrics['evaluate_episodes'], metrics['evaluate_test_envs_average_rewards'],
                     'evaluate_test_rewards', results_dir)
            lineplot(np.asarray(metrics['steps'])[np.asarray(metrics['evaluate_episodes']) - 1],
                     metrics['evaluate_test_envs_average_rewards'], 'evaluate_test_rewards_steps',
                     results_dir, xaxis='step')
            writer.add_scalar('evaluate_test_envs_all_rewards',
                              np.mean(average_evaluate_test_rewards),
                              metrics['steps'][-1])
        metrics['test_rewards'].append(average_evaluate_rewards.tolist())
        # metrics['test_rewards'].append(total_rewards)
        lineplot(metrics['evaluate_episodes'], metrics['test_rewards'],
                 'evaluate_rewards', results_dir)
        lineplot(np.asarray(metrics['steps'])[np.asarray(metrics['evaluate_episodes']) - 1],
                 metrics['test_rewards'], 'evaluate_rewards_steps', results_dir, xaxis='step')
        writer.add_scalar('evaluate_envs_all_rewards',
                          np.mean(average_evaluate_rewards),
                          metrics['steps'][-1])

        torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

        # Set models to train mode
        transition_model.train()
        observation_model.train()
        reward_model.train()
        encoder.train()
        actor_model.train()
        value_model.train()

        print("time cost:", time.time() - start_time)

    # Checkpoint models
    if episode % args.checkpoint_interval == 0:
        torch.save({
            'transition_model': transition_model.state_dict(),
            'observation_model': observation_model.state_dict(),
            'reward_model': reward_model.state_dict(),
            'encoder': encoder.state_dict(),
            'actor_model': actor_model.state_dict(),
            'value_model': value_model.state_dict(),
            'task_model': task_model.state_dict(),
            'model_optimizer': model_optimizer.state_dict(),
            'actor_optimizer': actor_optimizer.state_dict(),
            'value_optimizer': value_optimizer.state_dict(),
        },
            os.path.join(results_dir, 'models_%d.pth' % episode),
        )
        if args.checkpoint_experience:
            torch.save(
                D, os.path.join(results_dir, 'experience.pth')
            )  # Warning: will fail with MemoryError with large memory sizes

    if args.total_timesteps > 0 and metrics['steps'][-1] > args.total_timesteps:
        print("Current timesteps is", metrics['steps'][-1],
              " and total needed timesteps is", args.total_timesteps)
        break

# Close training environment
if args.env_type == 'dmc':
    for _ in range(train_environment_number):
        train_envs[_].close()

# save data to results_dir/data.txt
# path = os.path.join(results_dir, 'data.txt')
# filename = open(path, 'w')
# for k, v in metrics.items():
#     filename.write(k)
#     filename.write('\n')
#     for ii in v:
#         filename.write(str(ii) + ' ')
#     filename.write('\n')
# filename.close()
