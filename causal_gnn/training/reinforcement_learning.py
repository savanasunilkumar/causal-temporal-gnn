"""Reinforcement learning for sequential recommendations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """Experience replay buffer for RL."""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        """Add transition to buffer."""
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        """Sample batch of transitions."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNRecommender(nn.Module):
    """
    Deep Q-Network for recommendation.
    """

    def __init__(self, state_dim, action_dim, hidden_dims=[256, 128]):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Build network
        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        """
        Compute Q-values for all actions.

        Args:
            state: State representation [batch_size, state_dim]

        Returns:
            Q-values [batch_size, action_dim]
        """
        return self.network(state)

    def select_action(self, state, epsilon=0.1):
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            epsilon: Exploration rate

        Returns:
            Selected action
        """
        if random.random() < epsilon:
            # Explore: random action
            return random.randrange(self.action_dim)
        else:
            # Exploit: best action
            with torch.no_grad():
                q_values = self.forward(state)
                return q_values.argmax().item()


class ActorCritic(nn.Module):
    """
    Actor-Critic network for policy gradient methods.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state):
        """
        Forward pass through both actor and critic.

        Args:
            state: State tensor

        Returns:
            Action probabilities and state value
        """
        features = self.shared(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)

        return action_probs, state_value

    def select_action(self, state):
        """
        Sample action from policy.

        Args:
            state: Current state

        Returns:
            Action and log probability
        """
        action_probs, _ = self.forward(state)
        distribution = torch.distributions.Categorical(action_probs)
        action = distribution.sample()

        return action.item(), distribution.log_prob(action)


class DQNAgent:
    """
    DQN agent for recommendation.
    """

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 buffer_size=10000, batch_size=64, target_update_freq=100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Networks
        self.policy_net = DQNRecommender(state_dim, action_dim)
        self.target_net = DQNRecommender(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training step counter
        self.steps = 0

    def select_action(self, state, epsilon=0.1):
        """Select action using epsilon-greedy."""
        return self.policy_net.select_action(state, epsilon)

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Convert to tensors
        state_batch = torch.stack(batch.state)
        action_batch = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32)
        next_state_batch = torch.stack(batch.next_state)
        done_batch = torch.tensor(batch.done, dtype=torch.float32)

        # Compute Q(s, a)
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        # Compute loss
        loss = F.smooth_l1_loss(q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()


class A2CAgent:
    """
    Advantage Actor-Critic (A2C) agent.
    """

    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, entropy_coef=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        # Network
        self.network = ActorCritic(state_dim, action_dim)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        # Episode memory
        self.saved_log_probs = []
        self.rewards = []
        self.values = []

    def select_action(self, state):
        """Select action from policy."""
        action, log_prob = self.network.select_action(state)
        _, value = self.network(state)

        self.saved_log_probs.append(log_prob)
        self.values.append(value)

        return action

    def finish_episode(self):
        """Compute returns and update policy."""
        returns = []
        R = 0

        # Compute discounted returns
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        values = torch.stack(self.values).squeeze()
        log_probs = torch.stack(self.saved_log_probs)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute advantages
        advantages = returns - values.detach()

        # Actor loss (policy gradient)
        actor_loss = -(log_probs * advantages).mean()

        # Critic loss (value function)
        critic_loss = F.mse_loss(values, returns)

        # Entropy bonus for exploration
        entropy = -(torch.exp(log_probs) * log_probs).mean()

        # Total loss
        loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        # Clear memory
        self.saved_log_probs = []
        self.rewards = []
        self.values = []

        return loss.item()


class BanditAgent:
    """
    Multi-armed bandit for exploration-exploitation trade-off.
    """

    def __init__(self, num_arms, algorithm='ucb'):
        self.num_arms = num_arms
        self.algorithm = algorithm

        # Statistics
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.total_count = 0

    def select_arm(self, context=None):
        """
        Select arm based on algorithm.

        Args:
            context: Optional context for contextual bandits

        Returns:
            Selected arm index
        """
        if self.algorithm == 'ucb':
            return self._select_ucb()
        elif self.algorithm == 'thompson':
            return self._select_thompson()
        elif self.algorithm == 'epsilon_greedy':
            return self._select_epsilon_greedy()
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def _select_ucb(self, c=2.0):
        """Upper Confidence Bound selection."""
        if self.total_count < self.num_arms:
            # Try each arm at least once
            return self.total_count

        # Compute UCB scores
        exploration_bonus = c * np.sqrt(
            np.log(self.total_count) / (self.counts + 1e-5)
        )
        ucb_scores = self.values + exploration_bonus

        return np.argmax(ucb_scores)

    def _select_thompson(self):
        """Thompson sampling."""
        # Beta distribution parameters (successes, failures)
        alpha = self.values * self.counts + 1
        beta = (1 - self.values) * self.counts + 1

        # Sample from beta distributions
        samples = np.random.beta(alpha, beta)

        return np.argmax(samples)

    def _select_epsilon_greedy(self, epsilon=0.1):
        """Epsilon-greedy selection."""
        if np.random.random() < epsilon:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)

    def update(self, arm, reward):
        """
        Update statistics for selected arm.

        Args:
            arm: Selected arm index
            reward: Received reward
        """
        self.counts[arm] += 1
        self.total_count += 1

        # Update value (running average)
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward


class RecommendationEnvironment:
    """
    Reinforcement learning environment for recommendations.
    """

    def __init__(self, model, item_embeddings, user_history, reward_func='click'):
        self.model = model
        self.item_embeddings = item_embeddings
        self.user_history = user_history
        self.reward_func = reward_func

        self.current_user = None
        self.current_state = None

    def reset(self, user_idx):
        """
        Reset environment for new user.

        Args:
            user_idx: User index

        Returns:
            Initial state
        """
        self.current_user = user_idx

        # Get user embedding as initial state
        with torch.no_grad():
            _, user_emb, _ = self.model.forward(
                self.model.edge_index,
                self.model.edge_timestamps,
                self.model.time_indices
            )
            self.current_state = user_emb[user_idx]

        return self.current_state

    def step(self, action):
        """
        Take action (recommend item) and observe reward.

        Args:
            action: Item index to recommend

        Returns:
            next_state, reward, done, info
        """
        # Check if item was actually interacted with
        relevant_items = self.user_history.get(self.current_user, set())

        if self.reward_func == 'click':
            # Binary reward: 1 if clicked, 0 otherwise
            reward = 1.0 if action in relevant_items else 0.0
        elif self.reward_func == 'rating':
            # Could use actual rating if available
            reward = 1.0 if action in relevant_items else -0.1
        else:
            reward = 0.0

        # Update state (incorporate feedback)
        item_emb = self.item_embeddings[action]
        self.current_state = 0.9 * self.current_state + 0.1 * item_emb

        # Done after single recommendation (could extend to sessions)
        done = True

        info = {'action': action, 'user': self.current_user}

        return self.current_state, reward, done, info


def train_rl_recommender(env, agent, num_episodes=1000, epsilon_start=1.0,
                        epsilon_end=0.01, epsilon_decay=0.995):
    """
    Train RL agent for recommendations.

    Args:
        env: Recommendation environment
        agent: RL agent (DQN or A2C)
        num_episodes: Number of training episodes
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Decay rate for epsilon

    Returns:
        Training history
    """
    epsilon = epsilon_start
    rewards_history = []

    for episode in range(num_episodes):
        # Random user for this episode
        user_idx = np.random.randint(0, len(env.user_history))
        state = env.reset(user_idx)

        episode_reward = 0
        done = False

        while not done:
            # Select action
            if isinstance(agent, DQNAgent):
                action = agent.select_action(state, epsilon)
            else:  # A2C
                action = agent.select_action(state)

            # Take action
            next_state, reward, done, info = env.step(action)

            # Store transition
            if isinstance(agent, DQNAgent):
                agent.store_transition(state, action, reward, next_state, done)

                # Train
                loss = agent.train_step()
            else:  # A2C
                agent.rewards.append(reward)

            episode_reward += reward
            state = next_state

        # Finish episode for A2C
        if not isinstance(agent, DQNAgent):
            agent.finish_episode()

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        rewards_history.append(episode_reward)

        if episode % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.4f}, Epsilon: {epsilon:.4f}")

    return rewards_history
