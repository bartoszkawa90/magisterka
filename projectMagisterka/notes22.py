#!/usr/bin/env python3
import gym
import numpy as np
from typing import Optional, List
from collections import namedtuple
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
import torch.nn.utils as nn_utils


GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.05
BATCH_SIZE = 8
REWARD_MEAN_BOUND = 200
SEED = 42
CLIP_GRAD = 0.1
NUM_ITERATIONS = 80000000
REWARD_STEPS = 10


class ActorNet(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()

        self.hidden = nn.Linear(4, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, s):
        outs = self.hidden(s)
        outs = F.relu(outs)
        logits = self.output(outs)
        return logits


class ValueNet(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()

        self.hidden = nn.Linear(4, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, s):
        outs = self.hidden(s)
        outs = F.relu(outs)
        value = self.output(outs)
        return value


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
                # policy_out = net(state)[0]
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


def count_action_vals(exp_buff: List, net_policy, net_values) -> torch.FloatTensor:
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(exp_buff):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.new_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.new_state, copy=False))

    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = torch.FloatTensor(np.array(last_states, copy=False))
        last_vals_v = net_values(last_states_v)
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        last_vals_np *= GAMMA ** REWARD_STEPS
        rewards_np[not_done_idx] += last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np)
    return ref_vals_v


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
    # wandb.init(project="first-project")
    set_seed(SEED)
    net_agent = ActorNet(env.observation_space.shape[0])
    net_val = ValueNet(env.observation_space.shape[0])
    print(net_agent, '\n', net_val)

    agent = AgentPolicyGradient(env, GAMMA, REWARD_STEPS)

    optimizer_agent = optim.Adam(net_agent.parameters(), lr=LEARNING_RATE, eps=1e-3)
    optimizer_val = optim.Adam(net_val.parameters(), lr=LEARNING_RATE, eps=1e-3)

    total_rewards = []
    step_rewards = []
    exp_buffer = []
    step_idx = 0
    done_episodes = 0
    reward_sum = 0.0
    bs_smoothed = entropy = l_entropy = l_policy = l_total = None

    batch_states, batch_actions, batch_scales = [], [], []

    for step_idx in range(NUM_ITERATIONS):
        done_reward, exp = agent.make_a_step(net_agent)
        reward_sum += done_reward
        baseline = reward_sum / (step_idx + 1)
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        batch_scales.append(done_reward - baseline)
        step_rewards.append(exp.reward)
        exp_buffer.append(exp)

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

        optimizer_agent.zero_grad()
        optimizer_val.zero_grad()
        logits_v, values_v = net_agent(states_v), net_val(states_v)
        #TODO mozliwości, rewards, done_rewards, zdyskontowane albo nie
        # trzeba ogarnąć te wartości jakoś
        vref_vals = count_action_vals(exp_buffer, net_agent, net_val)
        loss_values_v = F.mse_loss(values_v, vref_vals) #TODO REQUIRED TO GET ACTION VALUES, CANT USE batch_scales, these all only rewards for steps

        log_prob_v = F.log_softmax(logits_v, dim=1)
        adv_v = vref_vals - values_v.detach()
        log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), batch_actions_t]
        loss_policy_v = -log_prob_actions_v.mean()

        prob_v = F.softmax(logits_v, dim=1)
        entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

        loss_policy_v.backward(retain_graph=True)

        # apply entropy and value gradients
        loss_v = entropy_loss_v + loss_values_v
        loss_v.backward()
        nn_utils.clip_grad_norm(net_agent.parameters(), CLIP_GRAD)
        nn_utils.clip_grad_norm(net_val.parameters(), CLIP_GRAD)

        optimizer_agent.step()
        optimizer_val.step()

        # wandb.log({"baseline": baseline, "baseline_entropy": entropy, 'loss_entropy': l_policy, 'loss_policy': l_total,
        #            'grad_l2': grad_means / grad_count, 'grad_max': grad_max, 'batch_scales': bs_smoothed})

        batch_states.clear()
        batch_actions.clear()
        batch_scales.clear()
        step_rewards.clear()
        exp_buffer.clear()

    # wandb.finish()
