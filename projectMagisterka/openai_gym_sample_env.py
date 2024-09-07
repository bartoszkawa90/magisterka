import gym

if __name__ == "__main__":
    env = gym.make("Blackjack-v1")
    print(f'Action space {env.action_space}')  # action space is Discrete(2), there are two action in "oczko" 1 to
    # take more cards and 0 to stop
    print(f'Observation space shape {env.observation_space.shape} with value {env.observation_space}')
    # przestrzen obserwacji to Tuple() 3 dyskretnych porzestrzeni, aktualną wartość kart gracza, wartość karty na stole i

    total_reward = 0.0
    total_steps = 0
    obs = env.reset()
    print(f'Observation after reset {obs} \n')

    while True:
        action = env.action_space.sample()
        print(f'Action for step {total_steps+1}  was {action}')
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        print(f'Observation after action {total_steps}  {obs}')
        if done:
            break
    print(f'\nEpisode finished in {total_steps} steps, total reword is {total_reward}'
          f' final obervation {obs}')