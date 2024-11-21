import gymnasium as gym
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from random import choices
from collections import namedtuple


GAMMA = 0.99
ENTROPY_BETA = 0.01
BATCH_SIZE = 10


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
    s, _ = env.reset()
    while 1:
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

        states_t = torch.FloatTensor(states)
        reward_t = torch.FloatTensor(rewards)
        actions_t = torch.LongTensor(actions)
        new_states_t = torch.FloatTensor(new_states)


        if done:
            R = 0
            reward_records.append(episode_reward)
            s, _ = env.reset()
            print(f' Episode {idx} with reward {episode_reward} and mean reward  {np.mean(reward_records[-50:])} ')
            episode_reward = 0
        else:
            R = critic(new_states_t[-1])

        R_values = []
        for ids, (state, action, reward) in enumerate(zip(reversed(states), reversed(actions), reversed(rewards))):
            if ids == 0:
                R_values.insert(0, reward + GAMMA*R)
            else:
                R_values.insert(0, reward + GAMMA*R_values[ids-1])

        R_values_t = torch.FloatTensor(R_values)

        # Simple but working optimization
        opt1.zero_grad()
        opt2.zero_grad()
        logits_v, value_v = actor(states_t), critic(states_t)
        loss_value_v = F.mse_loss(value_v.squeeze(-1), R_values_t)
        # apply value gradients
        loss_value_v.backward()

        log_prob_v = F.log_softmax(logits_v, dim=1)
        adv_v = R_values_t - value_v.detach()
        log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_t]
        loss_policy_v = -log_prob_actions_v.mean()

        # apply entropy gradients
        loss_policy_v.backward()
        # optimize
        opt1.step()
        opt2.step()

        states.clear()
        actions.clear()
        rewards.clear()
        new_states.clear()
        R_values.clear()

        idx += 1
        # Output total rewards in episode (max 500)
        print("Run episode {} with rewards {}".format(idx, np.mean(reward_records[-50:])), end="\r")

        # stop if reward mean > 475.0
        if np.mean(reward_records[-50:]) > 475.0 or idx > 2000000:
            break


if __name__ == "__main__":
    main()
