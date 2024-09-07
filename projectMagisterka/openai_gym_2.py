import gym


""" Sample code to show how random agent works with environment """

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    total_reward = 0.0
    total_steps = 0
    obs = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action) # one more element returned than book says, also method description
        # suggests that done flag is deprecated
        total_reward += reward
        total_steps += 1
        if done:
            break
    print(f'Episode finished in {total_steps} steps, total reword is {total_reward}')
