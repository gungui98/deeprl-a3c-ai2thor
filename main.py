""" Deep RL Algorithms for OpenAI Gym environments
"""

import argparse
import os
import sys

import gym
import numpy as np
import pandas as pd
import tensorflow as tf
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
    parser.add_argument('--type', type=str, default='A3C', help="Algorithm to train from { A3C }")
    parser.add_argument('--is_atari', dest='is_atari', action='store_true', help="Atari Environment")
    parser.add_argument('--is_ai2thor', dest='is_ai2thor', action='store_true', help="AI2Thor Environment")
    parser.add_argument('--with_PER', dest='with_per', action='store_true',
                        help="Use Prioritized Experience Replay (DDQN + PER)")
    parser.add_argument('--dueling', dest='dueling', action='store_true', help="Use a Dueling Architecture (DDQN)")
    #
    parser.add_argument('--nb_episodes', type=int, default=5000, help="Number of training episodes")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size (experience replay)")
    parser.add_argument('--consecutive_frames', type=int, default=4,
                        help="Number of consecutive frames (action repeat)")
    parser.add_argument('--training_interval', type=int, default=30, help="Network training frequency")
    parser.add_argument('--n_threads', type=int, default=1, help="Number of threads (A3C)")
    #
    parser.add_argument('--gather_stats', dest='gather_stats', action='store_true',
                        help="Compute Average reward per episode (slower)")
    parser.add_argument('--render', dest='render', action='store_true', help="Render environment while training")
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4', help="OpenAI Gym Environment")
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
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
    summary_writer = tf.summary.FileWriter(args.type + "/tensorboard_" + args.env)
    # Environment Initialization
    if args.is_ai2thor:
        config_dict = {'max_episode_length': 500}
        env = AI2ThorEnv(config_dict=config_dict)
        env.reset()
        state = env.reset()
        state_dim = state.shape
        action_dim = env.action_space.n
        env.close()
    elif (args.is_atari):
        # Atari Environment Wrapper
        env = AtariEnvironment(args)
        state_dim = env.get_state_size()
        action_dim = env.get_action_size()
        print(state_dim)
        print(action_dim)
    else:
        # Standard Environments
        env = Environment(gym.make(args.env), args.consecutive_frames)
        env.reset()
        state_dim = env.get_state_size()
        action_dim = gym.make(args.env).action_space.n

    algo = A3C(action_dim, state_dim, args.consecutive_frames, is_atari=args.is_atari, is_ai2thor=args.is_ai2thor)

    # Train
    stats = algo.train(env, args, summary_writer)

    # Export results to CSV
    if (args.gather_stats):
        df = pd.DataFrame(np.array(stats))
        df.to_csv(args.type + "/logs.csv", header=['Episode', 'Mean', 'Stddev'], float_format='%10.5f')

    # Save weights and close environments
    exp_dir = '{}/models/'.format(args.type)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    export_path = '{}{}_ENV_{}_NB_EP_{}_BS_{}'.format(exp_dir,
                                                      args.type,
                                                      args.env,
                                                      args.nb_episodes,
                                                      args.batch_size)

    algo.save_weights(export_path)
    env.close()


if __name__ == "__main__":
    main()
