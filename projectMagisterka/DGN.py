import gym
import torch as t
from torch import nn
import numpy as np
from torch import optim
import wandb


class DQNetwork(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(DQNetwork, self).__init__()
        self.net = nn.Sequential(

        )

    def forward(self, x):
        return self.net(x)

class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0


if __name__ == "__main__":
    env = gym.make('CartPole-v1')

    net = DQNetwork(input_shape=env.observation_space.n, output_shape=env.action_space.n)
    loss_fun = nn.SDG()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    # learning loop
    while True:

