#!/usr/bin/env python3
import gym
import collections
import wandb

ENV_NAME = "FrozenLake-v1"
GAMMA = 0.9
ALPHA = 0.2  # will be used as a learning factor during values actualization
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()[0]
        self.values = collections.defaultdict(float)

    def sample_env(self):
        """ Method does sample action in current env and collects states, action and reward """
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, _, _ = self.env.step(action)
        self.state = self.env.reset()[0] if is_done else new_state
        return old_state, action, reward, new_state

    def best_value_and_action(self, state):
        """Method used to choose best action and action value in current state"""
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def value_update(self, s, a, r, next_s):
        """Method chooses best action, counts Bellman equation and updated values table"""
        best_v, _ = self.best_value_and_action(next_s)
        new_v = r + GAMMA * best_v
        old_v = self.values[(s, a)]
        self.values[(s, a)] = old_v * (1-ALPHA) + new_v * ALPHA

    def play_episode(self, env):
        """"""
        total_reward = 0.0
        state = env.reset()[0]
        while True:
            _, action = self.best_value_and_action(state)
            new_state, reward, is_done, _, _ = env.step(action)
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    wandb.init(project="first-project")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        s, a, r, next_s = agent.sample_env()
        agent.value_update(s, a, r, next_s)

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        wandb.log({"Reward": reward, "iter": iter})
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (
                best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    wandb.finish()
