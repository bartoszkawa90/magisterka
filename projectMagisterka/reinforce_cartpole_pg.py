#!/usr/bin/env python3
import gym
import numpy as np
from typing import Optional
from collections import namedtuple
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os


GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.05
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
    for rew in reversed(rewards):
        sum_values = rew + gamma * sum_values
    return sum_values


class AgentPolicyGradient:
    def __init__(self, env, gamma, counter):
        self.env = env
        self.gamma = gamma
        self.counter = counter
        self.total_reward_ready = False
        self._reset()

    def _reset(self):
        self.state = env.reset(seed=SEED)[0]
        self.env.action_space.seed(SEED)
        self.total_reward = 0.0
        self.buffer = []
        self.done = False
        self.total_reward_ready = True

    def get_total_reward(self):
        if self.total_reward_ready:
            self.total_reward_ready = False
            return self.total_reward
        else:
            return None

    @torch.no_grad()
    def make_a_step(self, net: nn.Module):
        """Function to make agent steps with using possibilities
            Returns: Experience tuple and episode reward if episode has ended"""
        if not self.done:
            self.total_reward_ready = False
            self._reset()
            while True:
                state = torch.tensor(np.array(self.state))
                action_probs = F.softmax(net(state), dim=0)
                action = random.choices(list(range(self.env.action_space.n)), weights=action_probs)[0]
                # make a step in env
                new_state, reward, is_done, _, _ = self.env.step(action)
                self.total_reward += reward

                exp = Experience(self.state, action, reward, is_done, new_state)
                self.state = new_state
                self.buffer.append(exp)
                self.done = is_done
                if is_done:
                    break
        if self.done:
            temp_buff = [exp.reward for exp in self.buffer]
            temp = temp_buff[-1::-1][-10:]
            for _ in range(10):  # fill list of 1rewards with 0 when needed to
                if len(temp) < 10:
                    temp.append(0.0)
            discounted_reward = calc_qvals(temp, self.gamma)
            last_exp = self.buffer[0]
            self.buffer.pop(0)
            if len(self.buffer) == 0:
                self.done = False
            return discounted_reward, last_exp


def smooth(old: Optional[float], val: float, alpha: float = 0.95) -> float:
    if old is None:
        return val
    return old * alpha + (1-alpha)*val


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
    wandb.init(project="first-project")
    set_seed(SEED)
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

    for step_idx in range(800000):
        done_reward, exp = agent.make_a_step(net)
        reward_sum += done_reward
        baseline = reward_sum / (step_idx + 1)
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        batch_scales.append(done_reward - baseline)

        # handle new rewards
        new_rewards = agent.get_total_reward()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            # wandb.log({"mean_reward": mean_rewards, "reward": new_rewards})
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
                step_idx, reward, mean_rewards, done_episodes))
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
        '''Above the policy entropy is counted and stored in entropy_loss_v we consider this 
        value // subtract from loss value // so agent will not stuck in local minimum
        WE PREVENT AGENT FROM BEING TOO SURE OF HIS CHOICE'''

        loss_v.backward()
        optimizer.step()

        # calc KL-div
        '''Below we count divergence of Kullback-Leiblera for new and old policy, this value shows how much new policy is
        different from old one, we do not want any rapid changes so it needs to be a little bit smoothed
        Debug and observation purposes in this case, seems to have little effect and works fine without it
         IN THIS CASE'''
        new_logits_v = net(states_v)
        new_prob_v = F.softmax(new_logits_v, dim=1)
        kl_div_v = -((new_prob_v / prob_v).log() * prob_v).sum(dim=1).mean()

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

        wandb.log({"baseline": baseline, "baseline_entropy": entropy, 'loss_entropy': l_policy, 'loss_policy': l_total,
                   'grad_l2': grad_means / grad_count, 'grad_max': grad_max, 'batch_scales': bs_smoothed})

        batch_states.clear()
        batch_actions.clear()
        batch_scales.clear()

    wandb.finish()
