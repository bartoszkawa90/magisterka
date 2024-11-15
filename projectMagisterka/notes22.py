import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128
BATCH_SIZE = 8
MAX_EPISODES = 10000
ENTROPY_BETA = 0.01  # Entropy coefficient for exploration
ENV_NAME = "CartPole-v1"

# Actor Network
class ActorNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, output_dim),
        )

    def forward(self, x):
        return self.net(x)

# Critic Network
class CriticNet(nn.Module):
    def __init__(self, input_dim):
        super(CriticNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self, x):
        return self.net(x)

# Select an action using the actor network
def select_action(state, actor_net):
    state_tensor = torch.FloatTensor(state)
    logits = actor_net(state_tensor)
    logits = torch.clamp(logits, min=-20, max=20)  # Avoid extreme values
    action_probs = F.softmax(logits, dim=0)
    action = np.random.choice(len(action_probs), p=action_probs.detach().numpy())
    log_prob = torch.log(action_probs[action])
    return action, log_prob, action_probs

# Train the networks
def train():
    env = gym.make(ENV_NAME)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    # Initialize networks and optimizers
    actor_net = ActorNet(input_dim, output_dim)
    critic_net = CriticNet(input_dim)
    actor_optimizer = optim.Adam(actor_net.parameters(), lr=LEARNING_RATE)
    critic_optimizer = optim.Adam(critic_net.parameters(), lr=LEARNING_RATE)

    episode_rewards = []

    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()[0]
        states = []
        actions = []
        log_probs = []
        rewards = []
        entropies = []
        episode_reward = 0

        while True:
            # Select action
            action, log_prob, action_probs = select_action(state, actor_net)
            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10))  # Entropy calculation
            next_state, reward, done, truncated, _ = env.step(action)

            # Clip reward for stability
            reward = np.clip(reward, -1, 1)

            # Store step data
            states.append(state)
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)
            episode_reward += reward
            state = next_state

            # Update the network after every BATCH_SIZE steps or at the end of an episode
            if len(states) == BATCH_SIZE or done or truncated:
                # Compute discounted returns with bootstrapping
                next_value = 0 if done else critic_net(torch.FloatTensor(next_state)).item()
                discounted_returns = []
                R = next_value
                for r in reversed(rewards):
                    R = r + GAMMA * R
                    discounted_returns.insert(0, R)
                discounted_returns = torch.FloatTensor(discounted_returns)

                # Normalize returns
                if discounted_returns.std() > 0:
                    discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-5)

                # Convert batch data to tensors
                states_tensor = torch.FloatTensor(states)
                log_probs_tensor = torch.stack(log_probs)
                entropies_tensor = torch.stack(entropies)

                # Update Critic
                critic_optimizer.zero_grad()
                values = critic_net(states_tensor).squeeze()
                critic_loss = F.mse_loss(values, discounted_returns)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic_net.parameters(), max_norm=1.0)
                critic_optimizer.step()

                # Update Actor
                actor_optimizer.zero_grad()
                advantages = discounted_returns - values.detach()
                actor_loss = -(log_probs_tensor * advantages).mean() - ENTROPY_BETA * entropies_tensor.mean()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_net.parameters(), max_norm=1.0)
                actor_optimizer.step()

                # Reset batch
                states = []
                log_probs = []
                rewards = []
                entropies = []

            if done or truncated:
                break

        # Log results
        episode_rewards.append(episode_reward)
        print(f"Episode {episode}, Total Reward: {episode_reward}")

        # Check for solving condition
        if np.mean(episode_rewards[-100:]) > 475:
            print(f"Environment solved in {episode} episodes!")
            break

    env.close()

if __name__ == "__main__":
    train()
