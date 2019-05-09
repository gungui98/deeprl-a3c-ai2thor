""" Load and display pre-trained model in OpenAI Gym Environment
"""

import argparse
import os
import sys

import gym
from keras.backend.tensorflow_backend import set_session

from A3C.a3c import A3C
from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
from utils.atari_environment import AtariEnvironment
from utils.continuous_environments import Environment
from utils.networks import get_session

gym.logger.set_level(40)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    #
    parser.add_argument('--type', type=str, default='A3C', help="Algorithm to train from {A2C, A3C, DDQN, DDPG}")
    parser.add_argument('--is_atari', dest='is_atari', action='store_true', help="Atari Environment")
    parser.add_argument('--is_ai2thor', dest='is_ai2thor', action='store_true', help="AI2Thor Environment")
    #
    parser.add_argument('--model_path', type=str, help="Number of training episodes")
    parser.add_argument('--actor_path', type=str, help="Number of training episodes")
    parser.add_argument('--critic_path', type=str, help="Batch size (experience replay)")
    #
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4', help="OpenAI Gym Environment")
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--consecutive_frames', type=int, default=4,
                        help="Number of consecutive frames (action repeat)")

    #
    parser.set_defaults(render=False)
    return parser.parse_args(args)


def main(args=None):
    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    # Check if a GPU ID was set
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_session(get_session())

    # Environment Initialization
    if args.is_ai2thor:
        config_dict = {'max_episode_length': 2000}
        env = AI2ThorEnv(config_dict=config_dict)
        env.reset()
        state = env.reset()
        state_dim = state.shape
        action_dim = env.action_space.n
    elif (args.is_atari):
        # Atari Environment Wrapper
        env = AtariEnvironment(args)
        state_dim = env.get_state_size()
        action_dim = env.get_action_size()
    else:
        # Standard Environments
        env = Environment(gym.make(args.env), args.consecutive_frames)
        env.reset()
        state_dim = env.get_state_size()
        action_dim = gym.make(args.env).action_space.n

    algo = A3C(action_dim, state_dim, args.consecutive_frames, is_atari=args.is_atari, is_ai2thor=args.is_ai2thor)
    algo.load_weights(args.actor_path, args.critic_path)

    # Display agent
    old_state, time = env.reset(), 0
    while True:
        a = algo.policy_action(old_state)
        old_state, r, done, _ = env.step(a)
        time += 1
        if done:
            print('----- done, resetting env ----')
            env.reset()


if __name__ == "__main__":
    main()
