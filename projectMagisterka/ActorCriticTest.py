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


def calc_qvals(rewards, gamma):
    """Function calculates Q values (discounted summary reward for steps)
          We use reversed lists to make function more efficient"""
    sum_values = 0.0
    for rew in reversed(rewards):
        sum_values = rew + gamma * sum_values
    return sum_values


Experience = namedtuple('Experience', field_names=['state', 'action', 'reward'])
Episode = namedtuple('Episode', field_names=['reward', 'exp'])


def batch(env, net, batch_size=BATCH_SIZE):
    # global vars
    global FINAL_REWARD
    FINAL_REWARD = 0
    # vars
    steps = []
    # s, _ = env.reset()
    done = True
    while True:
        if done:
            s, _ = env.reset()
        a = pick_sample(s, net, env)
        s, r, term, trunc, _ = env.step(a)
        done = term  # or trunc
        steps.append(Experience(s, a, r)) # to s to prawdopodobnie jest nowe s czyli s' ale powwino sie brac stare
        if done:
            FINAL_REWARD = sum([e.reward for e in steps])
            # print(f'  FINAL REWARD {FINAL_REWARD} ')
            shift = 0
            rewards = 9*[0.0] + [e.reward for e in steps]
            # rewards = 9*[0.0] + [net_value(torch.FloatTensor(e.state)).detach().numpy()[0] for e in steps]
            for _, exp in zip(rewards, list(reversed(steps))):
                # print(f'   {list(reversed(rewards))[shift:batch_size+shift]}   for shift {shift}')
                yield Episode(calc_qvals(list(reversed(rewards))[shift:batch_size+shift], GAMMA), exp)
                shift += 1
            FINAL_REWARD = 0
            steps.clear()


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


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    set_seed()
    net_actor = ActorNet(env.observation_space.shape[0], env.action_space.n)
    net_value = ValueNet(env.observation_space.shape[0], env.action_space.n)
    reward_records = []
    opt1 = torch.optim.Adam(net_value.parameters(), lr=0.001)
    opt2 = torch.optim.Adam(net_actor.parameters(), lr=0.001)
    #TODO jeżeli aktor robi głupio przez 20 kroków a potem idzie dobrze bo przez pierwsze 20 mu się pofarciło więc trzeba
    # w jakichś zestawach typu po 10 będzie mądry w całości
    # Taki nieskończony horyzont wprowadza dużą losowość
    states_l = []
    actions_l = []
    rewards_l = []
    iter = 0
    reward_sum = 0.0
    for epi in batch(env, net_actor):
        iter += 1
        exp = epi.exp
        cum_rewards = epi.reward
        states_l.append(exp.state)
        actions_l.append(exp.action)
        reward_sum += cum_rewards
        baseline = reward_sum / (iter + 1)
        if cum_rewards == 1.0:
            reward_records.append(FINAL_REWARD)
        cum_rewards = cum_rewards - baseline
        rewards_l.append(cum_rewards)

        if len(rewards_l) < BATCH_SIZE:
            continue

        # Change observation variables to tensors
        states = torch.FloatTensor(states_l)
        cum_rewards = torch.FloatTensor(rewards_l)
        actions = torch.LongTensor(actions_l)


        # Optimize value loss (Critic)
        opt1.zero_grad()
        values = net_value(states)
        vf_loss = F.mse_loss(
            values.squeeze(dim=1),
            cum_rewards)
        vf_loss.sum().backward()
        opt1.step()

        # Optimize policy loss (Actor)
        with torch.no_grad():
            values = net_value(states)
        opt2.zero_grad()
        advantages = cum_rewards - values
        logits = net_actor(states)

        log_probs = F.log_softmax(logits)
        log_probs_actions = advantages * log_probs[range(len(actions)), actions]
        loss_policy = -log_probs_actions.mean()
        loss_policy.backward()
        opt2.step()


        # get full loss
        print(f'FINAL_REWARD {FINAL_REWARD} mean {np.mean(reward_records[-50:])} ')

        states_l.clear()
        actions_l.clear()
        rewards_l.clear()


        # Output total rewards in episode (max 500)
        print("Run episode {} with rewards {}".format(iter, FINAL_REWARD), end="\r")

        # stop if reward mean > 475.0
        if np.mean(reward_records[-50:]) > 475.0 or iter > 2000000:
            break

print("\nDone")
env.close()


