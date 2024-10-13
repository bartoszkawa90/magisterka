import gym
import torch
import torch as t
from torch import nn
import numpy as np
from torch import optim
import wandb
from collections import namedtuple, deque
import random
import os


LEARNING_RATE = 1e-4
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
REPLAY_START_SIZE = 10000
SYNC_TARGET_OBS = 1000
MEAN_REWARD_BOUND = 150  #400

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01

SEED = 42


Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)
        states, actions, rewards, dones, next_states = \
            zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states)


class DQNetwork(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(DQNetwork, self).__init__()
        # we want to use LeakyReLU which does not equal all negative values to 0 like ReLU, leaky relu has some slope
        # for negative values,
        # We want LeakyReLU because ReLU can make some neuron stop learning and LeakyReLU helps with that
        # https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
        self.net = nn.Sequential(
            nn.Linear(input_shape, 64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(64, output_shape)
        )

    def forward(self, x):
        return self.net(x)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset(seed=SEED)[0]
        self.env.action_space.seed(SEED)
        self.total_reward = 0.0

    @torch.no_grad()
    def make_a_step(self, net: nn.Module, epsilon: int = 0.0):
        """Make a single step random or not(possibility of random action is described by epsilon value)"""
        done_reward = None

        if random.random() < epsilon:
            # this way is random and does not work with seeding
            # action = env.action_space.sample()
            action = random.choice([0, 1])
        else:
            state = t.tensor(np.array(self.state))
            action_vals = net(state)
            action = int(t.argmax(action_vals))

        # make a step
        new_state, reward, is_done, _, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def complex_loss_function(batch, net, tgt_net):
    states, actions, rewards, dones, next_states = batch

    t_states = t.tensor(np.array(states))
    t_next_states = t.tensor(np.array(next_states))
    t_actions = t.tensor(actions)
    t_rewards = t.tensor(rewards)
    t_dones = t.BoolTensor(dones)

    # state_action_values = net(t_states)
    state_action_values = net(t_states).gather(1, t_actions.unsqueeze(-1)).squeeze(-1)
    # use torch.no_grad() because we are sure we don't want to calculate gradient, this way is more efficient
    with t.no_grad():
        next_state_values = tgt_net(t_next_states).max(1).values
        # if previous action ended episode (it was the last step in episode) we dont want to consider
        # next one because ot does not exist
        next_state_values[t_dones] = 0.0
        # we want to detach to avoid flow of gradients to neural network
        next_state_values = next_state_values.detach()

    # calculate (r + y * Q^(s, a)) and pass to MSELoss which finishes whole equation
    # MSELoss is (xn - yn)**2,  https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
    # expected_state_action_values = next_state_values * GAMMA + t_rewards
    expected_state_action_values = next_state_values * GAMMA + t_rewards
    return nn.MSELoss()(state_action_values, expected_state_action_values)


def set_seed(seed: int = 42) -> None:
    """Function which sets seed"""
    # env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


if __name__ == "__main__":
    # wandb.init(project="first-project")
    env = gym.make('CartPole-v1')
    # set seed
    set_seed(SEED)
    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    net = DQNetwork(input_shape=env.observation_space.shape[0], output_shape=env.action_space.n)
    target_net = DQNetwork(input_shape=env.observation_space.shape[0], output_shape=env.action_space.n)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    iter = 0
    # learning loop
    while True:
        iter += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START -
                      iter / EPSILON_DECAY_LAST_FRAME)

        reward = agent.make_a_step(net, epsilon)
        if reward is not None:  # episode is finished
            # append every new reward and check mean value of 100 latest rewards
            total_rewards.append(reward)
            m_reward = np.mean(total_rewards[-100:])
            print(f'For {iter} iteration: number of rewards {len(total_rewards)}, recent reward {reward}, '
                  f'mean reward {m_reward} and epsilon {epsilon}')
            if m_reward > MEAN_REWARD_BOUND:
                print(f"Good Good, time to finish ..., in {iter} iterations")
                break

        # skip loop if buffer is too small
        if len(buffer) < REPLAY_START_SIZE:
            continue
        # sync training and target networks every 1000 observations
        if iter % SYNC_TARGET_OBS == 0:
            target_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss = complex_loss_function(batch, net, target_net)
        loss.backward()
        optimizer.step()
    #     wandb.log({"mean reward of last 100": np.mean(total_rewards[-100:]),"reward after last step": reward,
    #                "epsilon": epsilon, "loss": loss})
    # wandb.finish()
