#!/usr/bin/env python3
import gym
import numpy as np
import random
from collections import namedtuple
from gym.spaces import Discrete
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 4
REWARD_MEAN_BOUND = 200
SEED = 42

#TODO remember that
# /Users/bartoszkawa/Desktop/astudia/magisterka/magisterka/projectMagisterka/venv/lib/python3.8/site-packages/ptan/experience.py
# this file was fixed, env.reset()[0] and done, _, _ added


Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


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


class Agent:
    def __init__(self, env):
        self.env = env
        self._reset()

    def _reset(self):
        self.state = env.reset(seed=SEED)[0]
        self.env.action_space.seed(SEED)
        self.total_reward = 0.0

    @torch.no_grad()
    def make_a_step(self, net: nn.Module):
        """Function to make agent steps with using possibilities
            Returns: Experience tuple and episode reward if episode has ended"""
        done_reward = None

        state = torch.tensor(np.array(self.state))
        action_probs = F.softmax(net(state), dim=0)
        if isinstance(self.env.action_space, Discrete):
            action = random.choices(list(range(self.env.action_space.n)), weights=action_probs)[0]

        # make a step
        new_state, reward, is_done, _, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward, exp


def calc_qvals(rewards):
    """Function calculates Q values (discounted summary reward for steps)
          We use reversed lists to make function more efficient"""
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))


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
    set_seed(SEED)
    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)
    agent = Agent(env)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    step_idx = 0
    done_episodes = 0

    batch_episodes = 0
    batch_states, batch_actions, batch_qvals = [], [], []
    cur_rewards = []

    step_idx = 0
    while True:
        done_reward, exp = agent.make_a_step(net)
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        cur_rewards.append(exp.reward)

        if done_reward:
            batch_qvals.extend(calc_qvals(cur_rewards))
            cur_rewards.clear()
            batch_episodes += 1
            done_episodes += 1
            # reward = new_rewards[0]
            reward = done_reward
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
                step_idx, reward, mean_rewards, done_episodes))
            if mean_rewards > REWARD_MEAN_BOUND:
                print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
                break

        step_idx += 1
        # skip rest of the loop if not enough episodes are stored in batch
        if batch_episodes < EPISODES_TO_TRAIN:
            continue

        optimizer.zero_grad()
        # change lists into tensors
        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_qvals_v = torch.FloatTensor(batch_qvals)

        # Descr:
        #   logits are raw unprocessed network outputs for some input data
        #   first we want to pass data to network and get output data
        #   2. we want to change out data into probabilities using softmax with log
        #   3. third we want to multiply Q values with log probabilities values for actions which was done for
        #   every episode, we can choose that by simpy passing batch_actions_t which has 1s and 0s which means what
        #   actions were done
        #   4. finally we get loss value by multiplying by -1 and counting sum // or mean value , works the same
        logits_v = net(states_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)
        log_prob_actions_v = batch_qvals_v * log_prob_v[range(len(batch_states)), batch_actions_t]
        # loss_v = -log_prob_actions_v.mean()
        loss_v = -torch.sum(log_prob_actions_v)

        loss_v.backward()
        optimizer.step()

        batch_episodes = 0
        batch_states.clear()
        batch_actions.clear()
        batch_qvals.clear()
