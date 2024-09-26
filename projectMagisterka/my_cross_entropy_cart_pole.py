# imports
from collections import namedtuple
import gym
import numpy as np
from torch import nn, FloatTensor, optim, LongTensor
import wandb


HIDDEN_N = 128
BATCH_SIZE = 16
PERCENTILE = 70


# network class (obs_n, hidden_n, act_n)
class Network(nn.Module):
    # class init
    def __init__(self, obs_num, hidden_n, act_num):
        super(Network, self).__init__()
        # network
        self.net = nn.Sequential(
            nn.Linear(obs_num, hidden_n),
            nn.ReLU(),
            nn.Linear(hidden_n, act_num)
        )
    # forward() method implementation
    def forward(self, x):
        return self.net(x)


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


# function for generating batches (env, net, batch_size) -> batch
def generate_batch(env, net, batch_size):
    # prepare data, reset env
    batch = []
    obs = env.reset()[0]
    reward_total = 0.0
    episode_steps = []
    sm = nn.Softmax(dim=1)
    # while loop for generating batches,
    while True:
        obs_v = FloatTensor([obs])
        act_probs_v = sm(net(obs_v))  # required because network does not return action posibilities on its own, we want it that way to use nn.CrossEntropyLoss
        act_probs = act_probs_v.data.numpy()[0]  # we want to get data in form of numpy array instead of tensor
        action = np.random.choice(len(act_probs), p=act_probs)
        new_obs, reward, is_done, _, _ = env.step(action)
        reward_total += reward
        episode_steps.append(EpisodeStep(obs, action))
        if is_done:
            batch.append(Episode(reward_total, episode_steps))
            episode_steps = []
            reward_total = 0.0
            new_obs = env.reset()[0]
            # return batch
            if len(batch) == batch_size:
                return batch
        obs = new_obs
       # OR
        # yield, return data when episode ended
        # reset env and variables


# function to filter episodes from batches (batch, percentile) -> return tensors
def filter_epizodes(batch, percentile):
    #  reward_bound and reward_mean/for featback purposes
    rewards = list(map(lambda r: r.reward, batch))
    reward_mean = np.mean(rewards)
    reward_bound = np.percentile(rewards, percentile)
    # filter episodes below bound
    train_obs = []
    train_act = []
    for reward, steps in batch:
        if reward >= reward_bound:
            _ = [train_obs.append(s.observation) for s in steps]
            _ = [train_act.append(s.action) for s in steps]
    return FloatTensor(train_obs), LongTensor(train_act), reward_mean, reward_bound


# main code with learning loop
if __name__ == "__main__":
    # add wandb instance to learning loop to log variables into wandb
    wandb.init(project="first-project")
    # prepare env, prepare net, loss_func and optimizer
    env = gym.make('CartPole-v1')
    net = Network(env.observation_space.shape[0], HIDDEN_N, env.action_space.n)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    # start learning loop with filtering every next batch
    # break or end loop when network works gooe enough
    iter = 0
    while True:
        batch = generate_batch(env, net, BATCH_SIZE)
        obs, act, r_mean, r_bound = filter_epizodes(batch, PERCENTILE)
        optimizer.zero_grad()
        net_actions = net(obs)
        loss = loss_func(net_actions, act)
        loss.backward()
        optimizer.step()
        wandb.log({"mean reward": r_mean, "loss": loss})
        print(f'iteration {iter} mean {r_mean} bound {r_bound}')
        if r_mean > 199:
            print('Finished')
            break
        iter += 1
    wandb.finish()
