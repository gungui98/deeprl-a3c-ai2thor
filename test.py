from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
N_EPISODES = 20
env = AI2ThorEnv()
max_episode_length = env.task.max_episode_length
for episode in range(N_EPISODES):
    state = env.reset()
    for step_num in range(max_episode_length):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        if done:
            break