from collections import namedtuple
import gym
import torch as t
from tensorboardX import SummaryWriter
import random
from torch import nn
import numpy as np
from torch import optim


HIDDEN_SIZE = 128  # number of hidden neurons
BATCH_SIZE = 100  # number of episodes
PERCENTILE = 30  # we do not consider 70% of worst episodes, we consider only 30% best episodes
GAMMA = 0.90  # stopa dyskontowa


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class DiscreteOneHotWrapper(gym.ObservationWrapper):
    """
    Wrapper for env because, FrozenLake provices numbers from 0 to 15 as obs and from 0 to 3 as actions,
    for actions it is fine but current network cannot work with this kind of observations so we have to change form
    of received obs by using wrapper, this way we can use same network
    """
    def __init__(self, env):
        super(DiscreteOneHotWrapper, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        shape = (env.observation_space.n, )
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, shape, dtype=np.float32)

    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net, batch_size):
    batch = []  # variable to store packs (lists of Episode class instances)
    episode_reward = 0.0  # reward counter
    episode_steps = []  # list of steps (class EpisodeSteps objects)
    obs = env.reset()[0]  # needed for newer version of gym, reset returns additional dict {}
    sm = nn.Softmax(dim=1)  # softmax layer will be used to get probabilities from data collected from net

    while True:
        obs_v = t.FloatTensor([obs])
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]  # we want to get data in form of numpy array instead of tensor
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _, _ = env.step(action)
        episode_reward += reward
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)
        if is_done:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()[0]  # needed for newer version of gym, reset returns additional dict {}
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    """Most important element of cross entropy method
       Function counts reward bound based on episodes, we need this to filter elite episodes"""
    filter_fun = lambda s: s.reward * (GAMMA ** len(s.steps))
    disc_rewards = list(map(filter_fun, batch))
    reward_bound = np.percentile(disc_rewards, percentile)  # numpy function which counts reward bound
    # smallest reward from (100-percentile) of best episodes
    train_obs = []
    train_act = []
    elite_batch = []
    for example, discounted_reward in zip(batch, disc_rewards):
        if discounted_reward > reward_bound:
            train_obs.extend(map(lambda step: step.observation, example.steps))
            train_act.extend(map(lambda step: step.action, example.steps))
            elite_batch.append(example)
    return elite_batch, train_obs, train_act, reward_bound


if __name__ == "__main__":
    # load environment
    env = DiscreteOneHotWrapper(gym.make('FrozenLake-v1')) # python somehow does not see toy_text lib in env is not opened
    # in this example environment is little different because we change ground to non-slippery, as a result networks learns faster
    env = gym.envs.toy_text.frozen_lake.FrozenLakeEnv(
        is_slippery=False)
    env.spec = gym.spec("FrozenLake-v1")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
    env = DiscreteOneHotWrapper(env)
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # choose network, objective function and optimizer
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.001)

    full_batch = []
    # start learning loop
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        reward_mean = float(np.mean(list(map(lambda s: s.reward, batch))))
        full_batch, obs, acts, reward_bound = filter_batch(full_batch + batch, PERCENTILE)
        if not full_batch:
            continue
        obs_v = t.FloatTensor(obs)
        acts_v = t.LongTensor(acts)
        full_batch = full_batch[-500:]

        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, rw_mean=%.3f, "
              "rw_bound=%.3f, batch=%d" % (
            iter_no, loss_v.item(), reward_mean,
            reward_bound, len(full_batch)))
        if reward_mean > 0.8:
            print("Solved!")
            break
    # writer.close()

