import gymnasium as gym
import os
import random
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from random import choices
from collections import namedtuple
import torch.nn.utils as nn_utils


GAMMA = 0.99
NUMBER_OF_EPISODES = 1500
BATCH_SIZE = 10
CLIP_GRAD = 0.1
ENTROPY_BETA = 0.01


class ActorNet(nn.Module):
    def __init__(self, input_size, n_actions, hidden_dim=128):
        super(ActorNet, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(input_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_actions)
            )

    def forward(self, x):
        return self.net(x)


class ValueNet(nn.Module):
    def __init__(self, input_size, n_actions, hidden_dim=128):
        super(ValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


# pick up action with above distribution policy_pi
def pick_sample(s, net, env):
        state = torch.tensor(np.array(s))
        action_probs = F.softmax(net(state), dim=0)
        action = choices(list(range(env.action_space.n)), weights=action_probs)[0]
        return action


def set_seed(seed: int = 42) -> None:
    """Function which sets seed"""
    # env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


Experience = namedtuple('Experience', field_names=['state', 'action', 'reward'])
Episode = namedtuple('Episode', field_names=['reward', 'exp'])


def main():
    '''
        It is important that we want to optimize every <fixed number> of steps cause in other way learning can lead to
        some random actions at start and good actions at the end and we do not want that
    '''
    env = gym.make("CartPole-v1")
    set_seed()
    actor = ActorNet(env.observation_space.shape[0], env.action_space.n)
    critic = ValueNet(env.observation_space.shape[0], env.action_space.n)
    reward_records = []
    opt1 = torch.optim.Adam(actor.parameters(), lr=0.001)
    opt2 = torch.optim.Adam(critic.parameters(), lr=0.001)
    states = []
    actions = []
    rewards = []
    new_states = []
    episode_reward = 0.0
    idx = 0
    new_s = 0.0
    while 1:
        s, _ = env.reset()
        a = pick_sample(s, actor, env)
        new_s, r, done, _, _ = env.step(a)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        new_states.append(new_s)
        s = new_s
        episode_reward += r

        if len(states) < BATCH_SIZE:
            continue

        if done:
            R = 0
            reward_records.append(episode_reward)
            episode_reward = 0
            s, _ = env.reset()
        else:
            R = critic(new_states[-1])

        R_values = len(state) * [0]
        for ids, state, action, reward in enumerate(reversed(states), reversed(actions), reversed(rewards)):
            if ids == 0:
                R_values.insert(0, reward + GAMMA*R)
            else:
                R_values.insert(0, reward + GAMMA*R_values[ids-1])

        # advantage = torch.FloatTensor(R_values) - critic(states).detach()

        states_t = torch.FloatTensor(states)
        reward_t = torch.FloatTensor(rewards)
        actions_t = torch.LongTensor(actions)
        R_values_t = torch.FloatTensor(R_values)

        # Optimize value loss (Critic)
        opt1.zero_grad()
        values = actor(states_t)
        vf_loss = F.mse_loss(
            values.squeeze(dim=1),
            R_values_t)
        vf_loss.sum().backward()
        opt1.step()

        # Optimize policy loss (Actor)
        with torch.no_grad():
            values = critic(states_t)
        opt2.zero_grad()
        advantages = R_values_t - values
        logits = actor(states_t)

        log_probs = F.log_softmax(logits)
        log_probs_actions = advantages * log_probs[range(len(actions_t)), actions_t]
        loss_policy = -log_probs_actions.mean()
        loss_policy.backward()
        opt2.step()

        states.clear()
        actions.clear()
        rewards.clear()
        new_states.clear()
        R_values.clear()

        # Output total rewards in episode (max 500)
        print("Run episode {} with rewards {}".format(iter, np.mean(reward_records[-50:])), end="\r")

        # stop if reward mean > 475.0
        if np.mean(reward_records[-50:]) > 475.0 or iter > 2000000:
            break


if __name__ == "__main__":
    main()
