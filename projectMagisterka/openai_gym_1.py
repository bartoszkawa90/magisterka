import gym


""" Sample code to show how Environment and Spaces in Gym lib works """

env = gym.make('CartPole-v0')

obs = env.reset()
print(f'observation after reset {obs}')

print(f'Action space {env.action_space}') # action spece is Discrete(2)
# so the platform moves left action=0 and right action=1

print(f'Observation space shape {env.observation_space.shape} with value {env.observation_space}')

first_step = env.step(0) # wybraliśmy aby przesunąć platforme w lewo, wybór akcji 0
print(f'First step {first_step}') # reward is 1 because in this environment it is always 1 and Agent aims for highest reward

print(f'Action space first sample {env.action_space.sample()}')
print(f'Action space second sample {env.action_space.sample()}')
print(f'Observation space first sample {env.observation_space.sample()}')
print(f'Observation space second sample {env.observation_space.sample()}')


