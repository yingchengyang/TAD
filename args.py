import argparse
from torch.nn import functional as F


# Hyperparameters
def get_parser(ipynb=False):
    parser = argparse.ArgumentParser(description='PlaNet or Dreamer')
    parser.add_argument('--algo', type=str, default='dreamer', help='planet or dreamer')
    # id: experiment id, default is args.algo
    parser.add_argument('--id', type=str, default="", help='Experiment ID')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
    # envs: all the env we hope to handle, separated by " ", for example
    # "walker-walk walker-stand walker-run"
    parser.add_argument('--envs', type=str, default='Pendulum-v0',
                        # choices=GYM_ENVS + CONTROL_SUITE_ENVS,
                        help='Gym/Control Suite environment', )

    # default: False
    parser.add_argument('--symbolic-env', default=False, help='Symbolic features')
    parser.add_argument('--save-video', default=False, type=bool)
    # 1000 for dmc, 200 for mujoco
    parser.add_argument('--max-episode-length', type=int, default=1000, metavar='T',
                        help='Max episode length')

    # Original implementation has an unlimited buffer size,
    # but 1 million is the max experience collected anyway
    parser.add_argument('--experience-size', type=int, default=1000000, metavar='D',
                        help='Experience replay size')
    parser.add_argument('--cnn-activation-function', type=str, default='relu',
                        choices=dir(F),
                        help='Model activation function for a convolution layer', )
    parser.add_argument('--dense-activation-function', type=str, default='elu',
                        choices=dir(F), help='Model activation function a dense layer', )

    # Note that the default encoder for visual observations outputs a 1024D vector;
    # for other embedding sizes an additional fully-connected layer is used
    parser.add_argument('--embedding-size', type=int, default=1024, metavar='E',
                        help='Observation embedding size')
    parser.add_argument('--hidden-size', type=int, default=200, metavar='H', help='Hidden size')
    parser.add_argument('--belief-size', type=int, default=200, metavar='H',
                        help='Belief/hidden size')
    parser.add_argument('--state-size', type=int, default=30, metavar='Z',
                        help='State/latent size')
    parser.add_argument('--action-repeat', type=int, default=2, metavar='R',
                        help='Action repeat')
    parser.add_argument('--action-noise', type=float, default=0.3, metavar='ε',
                        help='Action noise')
    # how many episodes we use
    parser.add_argument('--episodes', type=int, default=1000, metavar='E',
                        help='Total number of episodes')
    parser.add_argument('--total-timesteps', type=int, default=0,
                        help='Total timesteps for sampling timesteps')
    parser.add_argument('--seed-episodes', type=int, default=6, metavar='S',
                        help='Seed episodes')
    parser.add_argument('--collect-interval', type=int, default=100, metavar='C',
                        help='Collect interval')
    parser.add_argument('--batch-size', type=int, default=50, metavar='B',
                        help='Batch size')
    parser.add_argument('--chunk-size', type=int, default=50, metavar='L',
                        help='Chunk size')
    parser.add_argument('--worldmodel-LogProbLoss', action='store_true',
                        help='use LogProb loss for observation_model and '
                             'reward_model training', )
    parser.add_argument('--global-kl-beta', type=float, default=0, metavar='βg',
                        help='Global KL weight (0 to disable)')
    parser.add_argument('--free-nats', type=float, default=3, metavar='F',
                        help='Free nats')
    parser.add_argument('--bit-depth', type=int, default=5, metavar='B',
                        help='Image bit depth (quantisation)')

    # three learning rates
    parser.add_argument('--model_learning-rate', type=float, default=1e-3, metavar='α',
                        help='Learning rate')
    parser.add_argument('--actor_learning-rate', type=float, default=8e-5, metavar='α',
                        help='Learning rate')
    parser.add_argument('--value_learning-rate', type=float, default=8e-5, metavar='α',
                        help='Learning rate')
    parser.add_argument('--learning-rate-schedule', type=int, default=0, metavar='αS',
                        help='Linear learning rate schedule '
                             '(optimisation steps from 0 to final learning rate; '
                             '0 to disable)', )
    parser.add_argument('--adam-epsilon', type=float, default=1e-7, metavar='ε',
                        help='Adam optimizer epsilon value')

    # Note that original has a linear learning rate decay,
    # but it seems unlikely that this makes a significant difference
    parser.add_argument('--grad-clip-norm', type=float, default=100.0, metavar='C',
                        help='Gradient clipping norm')
    parser.add_argument('--planning-horizon', type=int, default=15, metavar='H',
                        help='Planning horizon distance')
    parser.add_argument('--discount', type=float, default=0.99, metavar='H',
                        help='discount')
    parser.add_argument('--disclam', type=float, default=0.95, metavar='H',
                        help='discount rate to compute return')
    parser.add_argument('--optimisation-iters', type=int, default=10, metavar='I',
                        help='Planning optimisation iterations')
    parser.add_argument('--candidates', type=int, default=1000, metavar='J',
                        help='Candidate samples per iteration')
    parser.add_argument('--top-candidates', type=int, default=100, metavar='K',
                        help='Number of top candidates to fit')
    parser.add_argument('--test-interval', type=int, default=25, metavar='I',
                        help='Test interval (episodes)')
    parser.add_argument('--test-episodes', type=int, default=10, metavar='E',
                        help='Number of test episodes')
    # origin is 50, I change it to 100
    parser.add_argument('--checkpoint-interval', type=int, default=100, metavar='I',
                        help='Checkpoint interval (episodes)')
    parser.add_argument('--checkpoint-experience', action='store_true',
                        help='Checkpoint experience replay')
    parser.add_argument('--models', type=str, default='', metavar='M',
                        help='Load model checkpoint')
    parser.add_argument('--experience-replay', type=str, default='', metavar='ER',
                        help='Load experience replay')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--device', default='cuda', type=str)

    # use p(r_t|s_t, h_t) or p(r_t|s_{t+1}, h_{t+1})
    parser.add_argument('--use-paper-reward', default=False, type=bool)
    # use q(s_{t+1}|h_{t+1}, o_{t+1}) or q(s_{t+1}|h_{t+1}, o_{t+1}, r_t)
    parser.add_argument('--use-encode-r-transition', default=False, type=bool)

    parser.add_argument('--reward-size', default=1.0, type=float)
    parser.add_argument('--sep-replay-buffer', default=False, type=bool)

    # test envs
    parser.add_argument('--test-envs', default=None, type=str)

    # task loss weight
    parser.add_argument('--task-weight', default=0.0, type=float)

    # whether use task vector in both training and testing
    parser.add_argument('--task-vector', default=False, type=bool)

    parser.add_argument('--task-loss-type', default='cross_entropy')
    parser.add_argument('--context-dim', default=0, type=int)
    parser.add_argument('--tau', default=0.1)

    # parameters from config
    parser.add_argument('--cfg', default=None, type=str)
    parser.add_argument('--not-use-enc', default=False, type=bool)
    parser.add_argument('--env-type', default='dmc', type=str) # dmc or mujoco
    # 1 for dmc, but for mujoco, we need larger sampled task per step
    parser.add_argument('--num_tasks_sample', default=1, type=int)
    if ipynb:
        return parser.parse_args([])
    else:
        return parser.parse_args()
