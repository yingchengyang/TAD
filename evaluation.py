import torch
from torch.distributions import Normal
from torchvision.utils import make_grid

from env import EnvBatcher


def update_belief_and_act(
        args_, env_, planner_, transition_model_, encoder_, belief_,
        posterior_state_, action_, observation_, explore_=False, rewards_=None,
        task_id_=None, task_model_=None,
):
    # Infer belief over current state q(s_t|o≤t,a<t) from the history
    # print("action size: ",action_.size()) torch.Size([1, 6])
    # Action and observation need extra time dimension
    # print('action', action_.shape) #[1,6]
    # print('rewards', rewards_.shape) #[1,1]
    # print('observation', encoder_(observation_).unsqueeze(dim=0).shape)
    if args_.env_type == 'mujoco' and args_.not_use_enc:
        encoder_observation_ = observation_.unsqueeze(dim=0)
    else:
        encoder_observation_ = encoder_(observation_).unsqueeze(dim=0)

    if task_id_ is not None:
        task_index_vector = torch.zeros(encoder_observation_.shape[0],
                                        encoder_observation_.shape[1],
                                        args_.train_environment_number,
                                        device=args_.device)
        task_index_vector[:, :, task_id_] = 1.0
        encoder_observation_ = torch.cat((encoder_observation_,
                                          task_index_vector),
                                         dim=2)

    # if not args_.use_encode_r_transition:
    #     belief_, _, _, _, posterior_state_, _, _ = \
    #         transition_model_(posterior_state_, action_.unsqueeze(dim=0),
    #                           belief_, encoder_observation_)
    # else:
    #     belief_, _, _, _, posterior_state_, _, _ = \
    #         transition_model_(posterior_state_, action_.unsqueeze(dim=0),
    #                           belief_, encoder_observation_,
    #                           rewards=rewards_)
    belief_, _, _, _, posterior_state_, _, _ = \
        transition_model_(posterior_state_, action_.unsqueeze(dim=0),
                          belief_, encoder_observation_,
                          rewards=rewards_)

    belief_, posterior_state_ = \
        belief_.squeeze(dim=0), posterior_state_.squeeze(dim=0)
    # Remove time dimension from belief/state

    if args_.algo == "dreamer" or args_.algo == "dreamer-tv" or args_.algo == "tad":
        action_ = planner_.get_action(belief_, posterior_state_,
                                      det=not explore_)
    elif args_.algo == "ctad":
        task_contexts = task_model_(belief_, posterior_state_)
        # print(belief_.shape)
        # print(posterior_state_.shape)
        # print(task_contexts.shape)
        # import pdb
        # pdb.set_trace()
        action_posterior_state_ = torch.cat([posterior_state_, task_contexts], dim=-1)
        action_ = planner_.get_action(belief_, action_posterior_state_,
                                      det=not explore_)
    elif args_.algo == "planet":
        # Get action from planner(q(s_t|o≤t,a<t), p)
        action_ = planner_(belief_, posterior_state_)
    else:
        raise Exception('Current algorithm is {}, but we only support algorithm including '
                        'planet, dreamer, tad, ctad, dreamer-tv'.format(args_.algo))

    if explore_:
        # add gaussian exploration noise on top of the sampled action:
        # action_ = action_ + args_.action_noise * torch.randn_like(action_)
        # add exploration noise ε ~ p(ε) to the action
        action_ = torch.clamp(Normal(action_, args_.action_noise).rsample(), -1, 1)

        # Perform environment step (action repeats handled internally)
    # we repeat actions in the env.py
    if args_.env_type == 'dmc':
        next_observation_, reward_, done_ = env_.step(action_.cpu()
                                                      if isinstance(env_, EnvBatcher)
                                                      else action_[0].cpu())
        info_ = None
    elif args_.env_type == 'mujoco':
        next_observation_, reward_, done_, info_ = env_.step(action_.cpu()
                                                             if isinstance(env_, EnvBatcher)
                                                             else action_[0].cpu())
    else:
        raise NotImplementedError
    return belief_, posterior_state_, action_, next_observation_, reward_, done_, info_


def sample_one_trajectory(args_, env_, planner_, transition_model_, encoder_,
                          explore_=False,
                          task_id_=None, test=False,
                          observation_model_=None,
                          task_model_=None,
                          env_type='dmc', env_name=None):
    with torch.no_grad():
        observation_, total_reward = env_.reset(), 0
        video_frames = []

        if test and env_type == 'dmc':
            belief_, posterior_state_, action_ = (
                torch.zeros(args_.test_episodes, args_.belief_size, device=args_.device),
                torch.zeros(args_.test_episodes, args_.state_size, device=args_.device),
                torch.zeros(args_.test_episodes, args_.action_size, device=args_.device),
            )
        else:
            belief_, posterior_state_, action_ = (
                torch.zeros(1, args_.belief_size, device=args_.device),
                torch.zeros(1, args_.state_size, device=args_.device),
                torch.zeros(1, env_.action_size, device=args_.device),
            )

        if env_type == 'mujoco':
            if env_name == "cheetah-dir" or env_name == "cheetah-vel":
                reward_forward = 0
                reward_ctrl = 0
            elif env_name == "ant-dir":
                reward_forward = 0
                reward_ctrl = 0
                reward_contact = 0
                reward_survive = 0
            elif env_name == "humanoid-dir":
                reward_linvel = 0
                reward_quadctrl = 0
                reward_alive = 0
                reward_impact = 0
            elif env_name == "ant-goal":
                goal_forward = 0
                reward_ctrl = 0
                reward_contact = 0
                reward_survive = 0
            else:
                raise NotImplementedError

        if not args_.use_encode_r_transition:
            current_rewards = None
        else:
            if test and env_type == 'dmc':
                current_rewards = args_.reward_size * torch.zeros(1, args_.test_episodes,
                                                                  1, device=args_.device)
            else:
                current_rewards = args_.reward_size * torch.zeros(1, 1, 1,
                                                                  device=args_.device)

        all_data = []

        for t in range(args_.max_episode_length // args_.action_repeat):
            # print("step",t)
            belief_, posterior_state_, action_, next_observation, reward, done, info = \
                update_belief_and_act(args_, env_, planner_, transition_model_,
                                      encoder_, belief_, posterior_state_, action_,
                                      observation_.to(device=args_.device),
                                      explore_=explore_,
                                      rewards_=current_rewards,
                                      task_id_=task_id_,
                                      task_model_=task_model_)

            if not args_.use_encode_r_transition:
                current_rewards = None
            else:
                if test and env_type == 'dmc':
                    current_rewards = args_.reward_size * reward.reshape(1, args_.test_episodes,
                                                                         1).to(device=args_.device)
                else:
                    current_rewards[0][0][0] = args_.reward_size * reward

            if test and env_type == 'dmc':
                total_reward += reward.numpy()
                if args_.save_video:
                    # Collect real vs. predicted frames for video
                    video_frames.append(
                        make_grid(
                            torch.cat([observation_,
                                       observation_model_(belief_, posterior_state_).cpu()],
                                      dim=3) + 0.5, nrow=5, ).numpy()
                    )
            else:
                total_reward += reward
                all_data.append((observation_, action_.cpu(), reward, done))

            if env_type == 'mujoco':
                if env_name == "cheetah-dir" or env_name == "cheetah-vel":
                    reward_forward += info["reward_forward"]
                    reward_ctrl += info["reward_ctrl"]
                elif env_name == "ant-dir":
                    reward_forward += info["reward_forward"]
                    reward_ctrl += info["reward_ctrl"]
                    reward_contact += info["reward_contact"]
                    reward_survive += info["reward_survive"]
                elif env_name == "humanoid-dir":
                    reward_linvel += info["reward_linvel"]
                    reward_quadctrl += info["reward_quadctrl"]
                    reward_alive += info["reward_alive"]
                    reward_impact += info["reward_impact"]
                elif env_name == "ant-goal":
                    goal_forward += info["goal_forward"]
                    reward_ctrl += info["reward_ctrl"]
                    reward_contact += info["reward_contact"]
                    reward_survive += info["reward_survive"]

            observation_ = next_observation
            if args_.render:
                env_.render()
            if test and env_type == 'dmc':
                if done.sum().item() == args_.test_episodes:
                    break
            else:
                if done:
                    if env_type == 'mujoco':
                        if env_name == "cheetah-dir" or env_name == "cheetah-vel":
                            print("reward_forward:", reward_forward, "reward_ctrl:", reward_ctrl)
                        elif env_name == "ant-dir":
                            print("reward_forward:", reward_forward, "reward_ctrl:", reward_ctrl,
                                  "reward_contact:", reward_contact, "reward_survive:", reward_survive)
                        elif env_name == "humanoid-dir":
                            print("reward_linvel:", reward_linvel, "reward_quadctrl:", reward_quadctrl,
                                  "reward_alive:", reward_alive, "reward_impact:", reward_impact)
                        elif env_name == "ant-goal":
                            print("goal_forward:", goal_forward, "reward_ctrl:", reward_ctrl,
                                  "reward_contact:", reward_contact, "reward_survive:", reward_survive)
                    break

        if test and env_type == 'mujoco':
            import numpy as np
            total_reward = np.array(total_reward).reshape(1,)

        # here return t+1 as t is counted from 0
        return t+1, total_reward, video_frames, all_data