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


# pick up action with above distribution policy_pi
def pick_sample(s, net, env):
    with torch.no_grad():
        state = torch.tensor(np.array(s))
        with torch.no_grad():
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


def batch(env, betch_len, net):
    # global vars
    global EPISODE_REWARD
    global FINAL_REWARD
    FINAL_REWARD = 0
    EPISODE_REWARD = 0
    # vars
    episodes = []
    i = 0
    s, _ = env.reset()
    while True:
        a = pick_sample(s, net, env)
        s, r, term, trunc, _ = env.step(a)
        done = term  # or trunc
        if done:
            r = 0
        episodes.append(Experience(s, a, r))
        EPISODE_REWARD += r
        if done:
            # print(f' --- Episode reward {EPISODE_REWARD} --- ')
            FINAL_REWARD = EPISODE_REWARD
            EPISODE_REWARD = 0
            s, _ = env.reset()
        if len(episodes) == betch_len:
            # print(f'Returning episode {i} with {len(episodes)} \n') # LOG
            i += 1
            yield episodes
            episodes.clear()


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
    opt1 = torch.optim.AdamW(net_value.parameters(), lr=0.001)
    opt2 = torch.optim.AdamW(net_actor.parameters(), lr=0.001)
    # for i in range(NUMBER_OF_EPISODES):

    # jeżeli aktor robi głupio przez 20 kroków a potem idzie dobrze bo przez pierwsze 20 mu się pofarciło więc trzeba
    # w jakichś zestawach typu po 10 będzie mądry w całości
    # Taki nieskończony horyzont wprowadza dużą losowość
    iter = 0
    for exp in batch(env, BATCH_SIZE, net_actor):
        iter += 1
        states = [e.state for e in exp]
        actions = [e.action for e in exp]
        rewards = [e.reward for e in exp]
        #
        # Run episode till done
        #
        # print(f'{i} episode {np.average(reward_records[-50:])} mean reward')
        # done = False
        # states = []
        # actions = []
        # rewards = []
        # s, _ = env.reset()
        # while not done:
        #     states.append(s.tolist())
        #     a = pick_sample(s, net_actor, env)
        #     s, r, term, trunc, _ = env.step(a)
        #     done = term #or trunc
        #     actions.append(a)
        #     rewards.append(r)

        # get cumulated discounter reward
        cum_rewards = np.zeros_like(rewards)
        rewards_copy = deepcopy(rewards)
        rewards_len = len(rewards)
        for i in reversed(range(rewards_len)):
            cum_rewards[rewards_len - 1 - i] = calc_qvals(rewards_copy, GAMMA)
            rewards_copy[i] = 0
        # print(f'Cumulated dicounted reward {cum_rewards} for iter {iter}') #LOG

        # Change observation variables to tensors
        states = torch.FloatTensor(states)
        cum_rewards = torch.FloatTensor(cum_rewards)
        actions = torch.LongTensor(actions)


        # internet version
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
        print(f'FINAL_REWARD ??? {FINAL_REWARD}')



        # book version
        # opt1.zero_grad()
        # opt2.zero_grad()
        # logits_v, value_v = net_actor(states), net_value(states)
        # loss_value_v = F.mse_loss(value_v.squeeze(-1), cum_rewards)
        #
        # log_prob_v = F.log_softmax(logits_v, dim=1)
        # adv_v = cum_rewards - value_v.detach()
        # log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions]
        # loss_policy_v = -log_prob_actions_v.mean()
        #
        # prob_v = F.softmax(logits_v, dim=1)
        # entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()
        #
        # # calculate policy gradients only
        # loss_policy_v.backward(retain_graph=True)
        # grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
        #                         for p in net_actor.parameters()
        #                         if p.grad is not None])
        #
        # # apply entropy and value gradients
        # loss_v = entropy_loss_v + loss_value_v
        # loss_v.backward()
        # nn_utils.clip_grad_norm_(net_actor.parameters(), CLIP_GRAD)
        # opt1.step()
        # opt2.step()
        # # get full loss
        # loss_v += loss_policy_v

        # Output total rewards in episode (max 500)
        print("Run episode {} with rewards {}".format(iter, FINAL_REWARD), end="\r")
        reward_records.append(FINAL_REWARD)

        # stop if reward mean > 475.0
        if np.mean(reward_records[-50:]) > 475.0 or iter > 20000:
            break

print("\nDone")
env.close()


