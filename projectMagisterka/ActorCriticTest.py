import gymnasium as gym
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class ActorNet(nn.Module):
#     def __init__(self, hidden_dim=16):
#         super().__init__()
#
#         self.hidden = nn.Linear(4, hidden_dim)
#         self.output = nn.Linear(hidden_dim, 2)
#
#     def forward(self, s):
#         outs = self.hidden(s)
#         outs = F.relu(outs)
#         logits = self.output(outs)
#         return logits
#
#
# class ValueNet(nn.Module):
#     def __init__(self, hidden_dim=16):
#         super().__init__()
#
#         self.hidden = nn.Linear(4, hidden_dim)
#         self.output = nn.Linear(hidden_dim, 1)
#
#     def forward(self, s):
#         outs = self.hidden(s)
#         outs = F.relu(outs)
#         value = self.output(outs)
#         return value


class ActorNet(nn.Module):
    def __init__(self, input_size, n_actions, hidden_dim=16):
        super(ActorNet, self).__init__()
        self.net = nn.Sequential(
                nn.Linear(input_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_actions)
            )

    def forward(self, x):
        return self.net(x)


class ValueNet(nn.Module):
    def __init__(self, input_size, n_actions, hidden_dim=16):
        super(ValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


gamma = 0.99

# pick up action with above distribution policy_pi
def pick_sample(s):
    with torch.no_grad():
        #   --> size : (1, 4)
        s_batch = np.expand_dims(s, axis=0)
        s_batch = torch.tensor(s_batch, dtype=torch.float).to(device)
        # Get logits from state
        #   --> size : (1, 2)
        logits = actor_func(s_batch)
        #   --> size : (2)
        logits = logits.squeeze(dim=0)
        # From logits to probabilities
        probs = F.softmax(logits, dim=-1)
        # Pick up action's sample
        a = torch.multinomial(probs, num_samples=1)
        # Return
        return a.tolist()[0]


def calc_qvals(rewards, gamma):
    """Function calculates Q values (discounted summary reward for steps)
          We use reversed lists to make function more efficient"""
    sum_values = 0.0
    for rew in reversed(rewards):
        sum_values = rew + gamma * sum_values
    return sum_values


env = gym.make("CartPole-v1")
actor_func = ActorNet(env.observation_space.shape[0], env.action_space.n).to(device)
value_func = ValueNet(env.observation_space.shape[0], env.action_space.n).to(device)
reward_records = []
opt1 = torch.optim.AdamW(value_func.parameters(), lr=0.001)
opt2 = torch.optim.AdamW(actor_func.parameters(), lr=0.001)
for i in range(1500):
    #
    # Run episode till done
    #
    print(f'{i} episode {np.average(reward_records[-50:])} mean reward')
    done = False
    states = []
    actions = []
    rewards = []
    s, _ = env.reset()
    while not done:
        states.append(s.tolist())
        a = pick_sample(s)
        s, r, term, trunc, _ = env.step(a)
        done = term #or trunc
        actions.append(a)
        rewards.append(r)

    #
    # Get cumulative rewards
    #
    # cum_rewards = np.zeros_like(rewards)
    # reward_len = len(rewards)
    # for j in reversed(range(reward_len)):
    #     cum_rewards[j] = rewards[j] + (cum_rewards[j+1]*gamma if j+1 < reward_len else 0)

    cum_rewards = np.zeros_like(rewards)
    rewards_copy = deepcopy(rewards)
    rewards_len = len(rewards)
    for i in reversed(range(rewards_len)):
        cum_rewards[rewards_len - 1 - i] = calc_qvals(rewards_copy, gamma)
        rewards_copy[i] = 0

    #
    # Train (optimize parameters)
    #

    # Optimize value loss (Critic)
    opt1.zero_grad()
    states = torch.tensor(states, dtype=torch.float).to(device)
    cum_rewards = torch.tensor(cum_rewards, dtype=torch.float).to(device)
    values = value_func(states)
    values = values.squeeze(dim=1)
    vf_loss = F.mse_loss(
        values,
        cum_rewards,
        reduction="none")
    vf_loss.sum().backward()
    opt1.step()

    # Optimize policy loss (Actor)
    with torch.no_grad():
        values = value_func(states)
    opt2.zero_grad()
    actions = torch.tensor(actions, dtype=torch.int64).to(device)
    advantages = cum_rewards - values
    logits = actor_func(states)
    log_probs = -F.cross_entropy(logits, actions, reduction="none")
    pi_loss = -log_probs * advantages
    pi_loss.sum().backward()
    opt2.step()

    # Output total rewards in episode (max 500)
    print("Run episode{} with rewards {}".format(i, sum(rewards)), end="\r")
    reward_records.append(sum(rewards))

    # stop if reward mean > 475.0
    if np.average(reward_records[-50:]) > 475.0:
        break

print("\nDone")
env.close()


