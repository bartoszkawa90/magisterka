from collections import namedtuple
import gym
import torch as t
from tensorboardX import SummaryWriter
from torch import nn
import numpy as np
from torch import optim


HIDDEN_SIZE = 128  # number of hidden neurons
BATCH_SIZE = 16  # number of episodes
PERCENTILE = 70  # we do not consider 70% of worst episodes, we consider only 30% best episodes


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
        # TODO obs_v = t.FloatTensor([obs])
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
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)  # numpy function which counts reward bound
    # smallest reward from (100-percentile) of best episodes
    reward_mean = float(np.mean(rewards))
    train_obs = []
    train_act = []
    for reward, steps in batch:
        if reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, steps))
        train_act.extend(map(lambda step: step.action, steps))
    train_obs_v = t.FloatTensor(train_obs)
    train_act_v = t.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean  # we change data back to torch tensors and return


if __name__ == "__main__":
    # load environment
    env = DiscreteOneHotWrapper(gym.make('FrozenLake-v1'))
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # choose network, objective function and optimizer
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)

    # start learning loop
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (
            iter_no, loss_v.item(), reward_m, reward_b))
        if reward_m > 199 or iter_no > 200:
            print("Solved!")
            break
    # writer.close()

