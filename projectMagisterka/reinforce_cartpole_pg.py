#!/usr/bin/env python3
import gym
# import ptan
import numpy as np
from typing import Optional
from collections import namedtuple
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.spaces import Discrete
import random


GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 8
REWARD_MEAN_BOUND = 200
SEED = 42

REWARD_STEPS = 10


class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


def calc_qvals(rewards, gamma):
    """Function calculates Q values (discounted summary reward for steps)
          We use reversed lists to make function more efficient"""
    sum_values = 0.0
    for reward in reversed(rewards):
        sum_values = reward + gamma * sum_values
    return sum_values


class AgentPolicyGradient:
    def __init__(self, env, gamma, count):
        self.env = env
        self.counter_rewards = count
        self.counter = count
        self._reset()
        self.buffer = []
        self.gamma = gamma
        self.done = True
        self.start_episode = True
        self.total_reward = 0.0

    def _reset(self):
        # print('Agent env reset')
        self.state = env.reset(seed=SEED)[0]
        self.env.action_space.seed(SEED)
        # self.total_reward = 0.0
        self.buffer = []

    @torch.no_grad()
    def step(self, net):
        state = torch.tensor(np.array(self.state))
        action_probs = F.softmax(net(state), dim=0)
        if isinstance(self.env.action_space, Discrete):
            action = random.choices(list(range(self.env.action_space.n)), weights=action_probs)[0]
        # make a step
        new_state, reward, is_done, _, _ = self.env.step(action)
        self.done = is_done
        self.total_reward += reward
        # experience
        exp = Experience(self.state, action, reward, is_done, new_state)
        self.state = new_state
        # self.buffer.append(exp)
        return exp

    def get_total_reward(self):
        if self.done:
            reward = self.total_reward
            self.total_reward = 0.0
            return reward
        else:
            return None

    @torch.no_grad()
    def make_a_step(self, net: nn.Module):
        """Function to make agent steps with using possibilities
            Returns: Experience tuple and episode reward if episode has ended"""
        state = torch.tensor(np.array(self.state))
        action_probs = F.softmax(net(state), dim=0)
        # if isinstance(self.env.action_space, Discrete): # check if obs space is discrete, it always is in this case
        action = random.choices(list(range(self.env.action_space.n)), weights=action_probs)[0]

        # make a step
        new_state, reward, is_done, _, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.state = new_state
        self.buffer.append(exp)
        if is_done:
            self._reset()
        last_10_discounted_reward = calc_qvals([e.reward for e in self.buffer[-10:]], self.gamma)
        # print(f' REWARD : {done_reward}')
        return last_10_discounted_reward, exp, is_done


def smooth(old: Optional[float], val: float, alpha: float = 0.95) -> float:
    if old is None:
        return val
    return old * alpha + (1-alpha)*val


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    # wandb.init(project="first-project")  #TODO add
    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)

    agent = AgentPolicyGradient(env, GAMMA, REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    step_rewards = []
    step_idx = 0
    done_episodes = 0
    reward_sum = 0.0
    bs_smoothed = entropy = l_entropy = l_policy = l_total = None

    batch_states, batch_actions, batch_scales = [], [], []

    for step_idx in range(8000000): # pętla tak na oko żeby coś chodziło
        done_reward, exp, done = agent.make_a_step(net)
        # print(f'--- REWARD : {done_reward}  ACTION {exp.action} STATE : {exp.state}---')
        reward_sum += done_reward
        baseline = reward_sum / (step_idx + 1)
        # writer.add_scalar("baseline", baseline, step_idx)
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        batch_scales.append(done_reward - baseline)

        # handle new rewards
        new_rewards = agent.get_total_reward() if done else None
        if new_rewards:
            done_episodes += 1
            reward = new_rewards
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
                step_idx, reward, mean_rewards, done_episodes))
            # writer.add_scalar("reward", reward, step_idx)
            # writer.add_scalar("reward_100", mean_rewards, step_idx)
            # writer.add_scalar("episodes", done_episodes, step_idx)
            if mean_rewards > REWARD_MEAN_BOUND:
                print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
                break

        if len(batch_states) < BATCH_SIZE:
            continue

        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_scale_v = torch.FloatTensor(batch_scales)

        optimizer.zero_grad()
        logits_v = net(states_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)
        log_prob_actions_v = batch_scale_v * log_prob_v[range(BATCH_SIZE), batch_actions_t]
        loss_policy_v = -log_prob_actions_v.mean()

        prob_v = F.softmax(logits_v, dim=1)
        entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
        entropy_loss_v = -ENTROPY_BETA * entropy_v
        loss_v = loss_policy_v + entropy_loss_v
        # print(f'Policy loss {loss_policy_v} and entropy loss {entropy_loss_v} for {step_idx} step')
        # print(f'Loss {loss_v} for {step_idx} iteration !!!')

        loss_v.backward()
        optimizer.step()

        # calc KL-div
        new_logits_v = net(states_v)
        new_prob_v = F.softmax(new_logits_v, dim=1)
        kl_div_v = -((new_prob_v / prob_v).log() * prob_v).sum(dim=1).mean()
        # writer.add_scalar("kl", kl_div_v.item(), step_idx)

        grad_max = 0.0
        grad_means = 0.0
        grad_count = 0
        for p in net.parameters():
            grad_max = max(grad_max, p.grad.abs().max().item())
            grad_means += (p.grad ** 2).mean().sqrt().item()
            grad_count += 1

        bs_smoothed = smooth(bs_smoothed, np.mean(batch_scales))
        entropy = smooth(entropy, entropy_v.item())
        l_entropy = smooth(l_entropy, entropy_loss_v.item())
        l_policy = smooth(l_policy, loss_policy_v.item())
        l_total = smooth(l_total, loss_v.item())

        # writer.add_scalar("baseline", baseline, step_idx)
        # writer.add_scalar("baseline", entropy, step_idx)
        # writer.add_scalar("loss_entropy", l_entropy, step_idx)
        # writer.add_scalar("loss_policy", l_policy, step_idx)
        # writer.add_scalar("loss_total", l_total, step_idx)
        # writer.add_scalar("grad_l2", grad_means / grad_count, step_idx)
        # writer.add_scalar("grad_max", grad_max, step_idx)
        # writer.add_scalar("batch_scales", bs_smoothed, step_idx)

        # wandb.log({"baseline": baseline, "baseline": baseline, 'loss_entropy': loss_entropy, 'loss_policy': loss_policy,
        #            'grad_l2': grad_l2, 'grad_max': grad_max, 'batch_scales': batch_scales})

        batch_states.clear()
        batch_actions.clear()
        batch_scales.clear()

    # writer.close()
    # wandb.finish()
