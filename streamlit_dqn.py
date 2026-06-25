#!/usr/bin/env python3
"""
Complex Deep Q-Learning Streamlit Application with Real MmCows Data
====================================================================

Interactive visualization and training of DQN agents for heat stress management.
Supports both synthetic and real MmCows dataset.

Features:
- Double DQN, Dueling DQN, Prioritized Experience Replay, Noisy Networks
- Real MmCows data integration
- Real-time training visualization
- Comprehensive metrics and analysis

Run with: streamlit run streamlit_dqn.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import random
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
import os

# =============================================================================
# DATA LOADER FOR MMCOWS (Real Data)
# =============================================================================

class MmCowsDataLoader:
    """Load and preprocess real MmCows dataset for RL training."""

    def __init__(self, data_dir: str = "data/mmcows_real/sensor_data"):
        self.data_dir = Path(data_dir)
        self.main_data = self.data_dir / "main_data"
        self.data = None
        self.is_loaded = False

    def check_data_exists(self) -> bool:
        """Check if real MmCows data exists."""
        return (
            self.main_data.exists() and
            (self.main_data / "cbt").exists() and
            (self.main_data / "thi").exists()
        )

    def load(self) -> pd.DataFrame:
        """Load and merge real MmCows sensor data."""
        if not self.check_data_exists():
            return None

        try:
            # Load CBT data
            cbt_records = []
            cbt_dir = self.main_data / "cbt"
            for csv_file in sorted(cbt_dir.glob("C*.csv")):
                cow_id = int(csv_file.stem[1:])
                df = pd.read_csv(csv_file)
                df['cow_id'] = cow_id
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                df = df.rename(columns={'temperature_C': 'cbt'})
                cbt_records.append(df[['datetime', 'cow_id', 'cbt']])

            if not cbt_records:
                return None
            cbt_df = pd.concat(cbt_records, ignore_index=True)

            # Load THI data (average)
            thi_file = self.main_data / "thi" / "average.csv"
            if thi_file.exists():
                thi_df = pd.read_csv(thi_file)
                thi_df['datetime'] = pd.to_datetime(thi_df['timestamp'], unit='s')
                thi_df = thi_df.rename(columns={'temperature_F': 'temperature', 'humidity_per': 'humidity', 'THI': 'thi'})
            else:
                return None

            # Load milk data
            milk_records = []
            milk_dir = self.main_data / "milk"
            for csv_file in sorted(milk_dir.glob("C*.csv")):
                cow_id = int(csv_file.stem[1:])
                df = pd.read_csv(csv_file)
                df['cow_id'] = cow_id
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                df = df.rename(columns={'milk_weight_kg': 'milk_yield'})
                milk_records.append(df[['datetime', 'cow_id', 'milk_yield']])
            milk_df = pd.concat(milk_records, ignore_index=True) if milk_records else None

            # Resample CBT to 30min
            cbt_resampled = []
            for cow_id in cbt_df['cow_id'].unique():
                cow_data = cbt_df[cbt_df['cow_id'] == cow_id].set_index('datetime')
                cow_data = cow_data.resample('30min').mean()
                cow_data['cow_id'] = cow_id
                cbt_resampled.append(cow_data.reset_index())
            cbt_df = pd.concat(cbt_resampled, ignore_index=True)

            # Resample THI
            thi_df = thi_df.set_index('datetime').resample('30min').mean().reset_index()

            # Merge CBT with THI
            merged = pd.merge_asof(
                cbt_df.sort_values('datetime'),
                thi_df[['datetime', 'temperature', 'humidity', 'thi']].sort_values('datetime'),
                on='datetime',
                direction='nearest'
            )

            # Add milk yield
            if milk_df is not None:
                milk_df['date'] = milk_df['datetime'].dt.date
                merged['date'] = merged['datetime'].dt.date
                milk_daily = milk_df.groupby(['cow_id', 'date'])['milk_yield'].mean().reset_index()
                merged = merged.merge(milk_daily, on=['cow_id', 'date'], how='left')
                merged['milk_yield'] = merged.groupby('cow_id')['milk_yield'].ffill().bfill()
                merged = merged.drop(columns=['date'])
            else:
                merged['milk_yield'] = 30.0

            # Add hour and defaults
            merged['hour'] = merged['datetime'].dt.hour
            merged['lying_ratio'] = 0.5
            merged['activity'] = 0.3

            # Clean up
            merged = merged.dropna(subset=['cbt', 'thi'])
            merged['milk_yield'] = merged['milk_yield'].fillna(30.0)

            self.data = merged
            self.is_loaded = True
            return self.data

        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if self.data is None:
            return {}

        return {
            'n_samples': len(self.data),
            'n_cows': self.data['cow_id'].nunique(),
            'thi_range': (self.data['thi'].min(), self.data['thi'].max()),
            'cbt_range': (self.data['cbt'].min(), self.data['cbt'].max()),
            'milk_range': (self.data['milk_yield'].min(), self.data['milk_yield'].max()),
            'heat_stress_ratio': (self.data['thi'] > 72).mean(),
            'date_range': (self.data['datetime'].min(), self.data['datetime'].max()) if 'datetime' in self.data.columns else None
        }


# =============================================================================
# HEAT STRESS ENVIRONMENT WITH REAL DATA SUPPORT
# =============================================================================

class HeatStressEnv(gym.Env):
    """Heat Stress Environment supporting both synthetic and real MmCows data."""

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        episode_length: int = 48,
        reward_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        self.data = data
        self.episode_length = episode_length
        self.use_real_data = data is not None and len(data) > 0

        self.reward_weights = reward_weights or {
            'milk': 0.4,
            'comfort': 0.4,
            'energy': 0.2
        }

        # State: [THI, CBT, hour_sin, hour_cos, lying_ratio, activity, milk_prev]
        self.observation_space = spaces.Box(
            low=np.array([50, 37.5, -1, -1, 0, 0, 0]),
            high=np.array([95, 42, 1, 1, 1, 1, 50]),
            dtype=np.float32
        )

        # Actions: 0=none, 1=fans, 2=fans+sprinklers, 3=water_spray
        self.action_space = spaces.Discrete(4)

        # Cooling effects and costs
        self.cooling_effects = {0: 0.0, 1: 0.2, 2: 0.6, 3: 0.3}
        self.energy_costs = {0: 0.0, 1: 0.2, 2: 0.8, 3: 0.4}

        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.step_count = 0

        if self.use_real_data:
            # Select random cow and starting point
            cow_ids = self.data['cow_id'].unique()
            self.current_cow = np.random.choice(cow_ids)
            cow_data = self.data[self.data['cow_id'] == self.current_cow]

            max_start = max(0, len(cow_data) - self.episode_length)
            if max_start > 0:
                start_idx = np.random.randint(0, max_start)
            else:
                start_idx = 0

            self.episode_data = cow_data.iloc[start_idx:start_idx + self.episode_length].reset_index(drop=True)
            self._load_state_from_data(0)
        else:
            self._generate_synthetic_state()

        return self._get_obs(), {'cow_id': getattr(self, 'current_cow', None)}

    def _load_state_from_data(self, idx: int):
        """Load state from real data."""
        if idx < len(self.episode_data):
            row = self.episode_data.iloc[idx]
            self.thi = float(row.get('thi', 70))
            self.cbt = float(row.get('cbt', 38.5))
            self.hour = int(row.get('hour', 12))
            self.lying_ratio = float(row.get('lying_ratio', 0.5))
            self.activity = float(row.get('activity', 0.3))
            self.milk = float(row.get('milk_yield', 28))
        else:
            self._generate_synthetic_state()

    def _generate_synthetic_state(self):
        """Generate synthetic state."""
        self.hour = np.random.randint(0, 24)
        self.thi = 65 + 15 * np.sin(np.pi * (self.hour - 6) / 12) + np.random.normal(0, 3)
        self.thi = np.clip(self.thi, 50, 95)
        self.cbt = 38.5 + 0.03 * max(0, self.thi - 68) + np.random.normal(0, 0.15)
        self.cbt = np.clip(self.cbt, 37.5, 42)
        self.lying_ratio = 0.6 - 0.01 * max(0, self.thi - 68) + np.random.normal(0, 0.1)
        self.lying_ratio = np.clip(self.lying_ratio, 0, 1)
        self.activity = 0.4 if 6 <= self.hour <= 20 else 0.2
        self.activity = np.clip(self.activity + np.random.normal(0, 0.1), 0, 1)
        self.milk = 28 - 0.3 * max(0, self.thi - 72) + np.random.normal(0, 2)
        self.milk = np.clip(self.milk, 0, 50)

    def _get_obs(self):
        hour_sin = np.sin(2 * np.pi * self.hour / 24)
        hour_cos = np.cos(2 * np.pi * self.hour / 24)
        return np.array([
            self.thi, self.cbt, hour_sin, hour_cos,
            self.lying_ratio, self.activity, self.milk
        ], dtype=np.float32)

    def step(self, action):
        self.step_count += 1

        # Store previous values
        cbt_prev = self.cbt
        milk_prev = self.milk

        # Apply cooling effect
        cooling = self.cooling_effects[action]
        self.cbt = max(37.5, self.cbt - cooling)

        # Calculate comfort (0-1)
        thi_score = 1.0 - min(1.0, max(0, self.thi - 68) / 22)
        cbt_score = 1.0 - min(1.0, abs(self.cbt - 38.5) / 2.5)
        comfort = 0.5 * thi_score + 0.5 * cbt_score

        # Milk production effect
        if self.cbt > 39.5:
            milk_loss = -0.4 * (self.cbt - 39.5)
        elif self.cbt < 39.0:
            milk_loss = 0.1 * comfort
        else:
            milk_loss = 0

        self.milk = np.clip(self.milk + milk_loss, 0, 50)

        # Calculate reward
        milk_reward = (self.milk - milk_prev) / 5.0  # Normalized
        comfort_reward = comfort
        energy_penalty = self.energy_costs[action]

        reward = (
            self.reward_weights['milk'] * milk_reward +
            self.reward_weights['comfort'] * comfort_reward -
            self.reward_weights['energy'] * energy_penalty
        )

        # Bonus for good heat management
        if self.thi > 72 and self.cbt < 39.2:
            reward += 0.15

        # Update state for next step
        if self.use_real_data and self.step_count < len(self.episode_data):
            self._load_state_from_data(self.step_count)
            # Keep CBT modified by action
            self.cbt = max(37.5, self.cbt - cooling * 0.5)
        else:
            self._transition_synthetic(action)

        terminated = self.step_count >= self.episode_length
        truncated = False

        info = {
            'thi': self.thi,
            'cbt': self.cbt,
            'cbt_change': self.cbt - cbt_prev,
            'milk': self.milk,
            'milk_change': self.milk - milk_prev,
            'comfort': comfort,
            'energy_cost': energy_penalty,
            'action_name': ['none', 'fans', 'fans+spray', 'spray'][action]
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _transition_synthetic(self, action):
        """Transition to next synthetic state."""
        self.hour = (self.hour + 1) % 24

        # THI follows daily pattern
        base_thi = 65 + 15 * np.sin(np.pi * (self.hour - 6) / 12)
        self.thi = base_thi + np.random.normal(0, 2)
        self.thi = np.clip(self.thi, 50, 95)

        # CBT affected by THI and cooling
        heat_effect = 0.02 * max(0, self.thi - 68)
        self.cbt = self.cbt + heat_effect + np.random.normal(0, 0.05)
        self.cbt = np.clip(self.cbt, 37.5, 42)

        # Behavior changes
        self.lying_ratio = 0.6 - 0.01 * max(0, self.thi - 68) + np.random.normal(0, 0.05)
        self.lying_ratio = np.clip(self.lying_ratio, 0, 1)

        self.activity = 0.4 if 6 <= self.hour <= 20 else 0.2
        self.activity = np.clip(self.activity + np.random.normal(0, 0.05), 0, 1)


# =============================================================================
# NEURAL NETWORK ARCHITECTURES
# =============================================================================

class DQNNetwork(nn.Module):
    """Standard DQN Network."""

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int] = None):
        super().__init__()
        hidden_sizes = hidden_sizes or [128, 128]

        layers = []
        prev_size = state_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, action_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DuelingDQNNetwork(nn.Module):
    """Dueling DQN with separate value and advantage streams."""

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int] = None):
        super().__init__()
        hidden_sizes = hidden_sizes or [128, 128]

        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.LayerNorm(hidden_sizes[0])
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], action_dim)
        )

    def forward(self, x):
        features = self.features(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values


class NoisyLinear(nn.Module):
    """Noisy Linear layer for exploration."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class NoisyDQNNetwork(nn.Module):
    """DQN with Noisy Networks for exploration."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )

        self.noisy1 = NoisyLinear(hidden_size, hidden_size)
        self.noisy2 = NoisyLinear(hidden_size, action_dim)

    def forward(self, x):
        features = self.features(x)
        x = F.relu(self.noisy1(features))
        return self.noisy2(x)

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()


# =============================================================================
# EXPERIENCE REPLAY BUFFERS
# =============================================================================

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """Standard Experience Replay Buffer."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer."""

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def push(self, *args):
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(*args))
        else:
            self.buffer[self.position] = Transition(*args)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]

        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# DQN AGENT
# =============================================================================

@dataclass
class DQNConfig:
    """Configuration for DQN Agent."""
    state_dim: int = 7
    action_dim: int = 4
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    buffer_size: int = 10000
    batch_size: int = 64
    target_update_freq: int = 10
    use_double_dqn: bool = True
    use_dueling: bool = False
    use_per: bool = False
    use_noisy: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 10000


class DQNAgent:
    """Deep Q-Network Agent with multiple enhancements."""

    def __init__(self, config: DQNConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create networks
        if config.use_noisy:
            self.policy_net = NoisyDQNNetwork(
                config.state_dim, config.action_dim
            ).to(self.device)
            self.target_net = NoisyDQNNetwork(
                config.state_dim, config.action_dim
            ).to(self.device)
        elif config.use_dueling:
            self.policy_net = DuelingDQNNetwork(
                config.state_dim, config.action_dim, config.hidden_sizes
            ).to(self.device)
            self.target_net = DuelingDQNNetwork(
                config.state_dim, config.action_dim, config.hidden_sizes
            ).to(self.device)
        else:
            self.policy_net = DQNNetwork(
                config.state_dim, config.action_dim, config.hidden_sizes
            ).to(self.device)
            self.target_net = DQNNetwork(
                config.state_dim, config.action_dim, config.hidden_sizes
            ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)

        if config.use_per:
            self.memory = PrioritizedReplayBuffer(config.buffer_size, config.per_alpha)
        else:
            self.memory = ReplayBuffer(config.buffer_size)

        self.epsilon = config.epsilon_start
        self.steps_done = 0
        self.losses = []
        self.q_values = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if self.config.use_noisy:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                return q_values.argmax().item()

        if training and random.random() < self.epsilon:
            return random.randint(0, self.config.action_dim - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update(self) -> Optional[float]:
        if len(self.memory) < self.config.batch_size:
            return None

        if self.config.use_per:
            beta = min(1.0, self.config.per_beta_start +
                      self.steps_done * (1.0 - self.config.per_beta_start) / self.config.per_beta_frames)
            transitions, indices, weights = self.memory.sample(self.config.batch_size, beta)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            transitions = self.memory.sample(self.config.batch_size)
            weights = torch.ones(self.config.batch_size).to(self.device)

        batch = Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)

        current_q = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        with torch.no_grad():
            if self.config.use_double_dqn:
                next_actions = self.policy_net(next_state_batch).argmax(1, keepdim=True)
                next_q = self.target_net(next_state_batch).gather(1, next_actions)
            else:
                next_q = self.target_net(next_state_batch).max(1, keepdim=True)[0]

            target_q = reward_batch.unsqueeze(1) + (1 - done_batch.unsqueeze(1)) * self.config.gamma * next_q

        td_errors = (current_q - target_q).abs().detach().cpu().numpy()
        loss = (weights.unsqueeze(1) * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        if self.config.use_per:
            self.memory.update_priorities(indices, td_errors.squeeze() + 1e-6)

        if self.config.use_noisy:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()

        loss_value = loss.item()
        self.losses.append(loss_value)
        self.q_values.append(current_q.mean().item())

        return loss_value

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_net(state_t).cpu().numpy()[0]

    def save(self, path: str):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'config': self.config
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_episode(agent: DQNAgent, env: gym.Env, episode: int) -> dict:
    """Train for one episode."""
    state, info = env.reset()
    total_reward = 0
    steps = 0
    losses = []

    episode_data = {
        'states': [], 'actions': [], 'rewards': [], 'q_values': [],
        'thi': [], 'cbt': [], 'milk': [], 'comfort': []
    }

    while True:
        action = agent.select_action(state, training=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.store_transition(state, action, reward, next_state, done)

        episode_data['states'].append(state)
        episode_data['actions'].append(action)
        episode_data['rewards'].append(reward)
        episode_data['q_values'].append(agent.get_q_values(state))
        episode_data['thi'].append(info.get('thi', 70))
        episode_data['cbt'].append(info.get('cbt', 38.5))
        episode_data['milk'].append(info.get('milk', 28))
        episode_data['comfort'].append(info.get('comfort', 0.5))

        loss = agent.update()
        if loss is not None:
            losses.append(loss)

        total_reward += reward
        steps += 1
        state = next_state
        agent.steps_done += 1

        if done:
            break

    if episode % agent.config.target_update_freq == 0:
        agent.update_target_network()

    agent.decay_epsilon()

    return {
        'episode': episode,
        'total_reward': total_reward,
        'steps': steps,
        'avg_loss': np.mean(losses) if losses else 0,
        'epsilon': agent.epsilon,
        'episode_data': episode_data,
        'avg_comfort': np.mean(episode_data['comfort']),
        'avg_cbt': np.mean(episode_data['cbt']),
        'final_milk': episode_data['milk'][-1] if episode_data['milk'] else 0
    }


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_training_plots(metrics_history: List[dict]) -> go.Figure:
    """Create training progress plots."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Episode Rewards', 'Training Loss', 'Epsilon Decay', 'Comfort & CBT'),
        vertical_spacing=0.12
    )

    episodes = [m['episode'] for m in metrics_history]
    rewards = [m['total_reward'] for m in metrics_history]
    losses = [m['avg_loss'] for m in metrics_history]
    epsilons = [m['epsilon'] for m in metrics_history]
    comforts = [m.get('avg_comfort', 0.5) for m in metrics_history]
    cbts = [m.get('avg_cbt', 38.5) for m in metrics_history]

    window = min(10, len(rewards))
    rewards_ma = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
    losses_ma = pd.Series(losses).rolling(window=window, min_periods=1).mean()

    # Rewards
    fig.add_trace(go.Scatter(x=episodes, y=rewards, mode='lines', name='Reward',
                             line=dict(color='lightblue', width=1), opacity=0.5), row=1, col=1)
    fig.add_trace(go.Scatter(x=episodes, y=rewards_ma, mode='lines', name='Reward (MA)',
                             line=dict(color='blue', width=2)), row=1, col=1)

    # Loss
    fig.add_trace(go.Scatter(x=episodes, y=losses, mode='lines', name='Loss',
                             line=dict(color='lightcoral', width=1), opacity=0.5), row=1, col=2)
    fig.add_trace(go.Scatter(x=episodes, y=losses_ma, mode='lines', name='Loss (MA)',
                             line=dict(color='red', width=2)), row=1, col=2)

    # Epsilon
    fig.add_trace(go.Scatter(x=episodes, y=epsilons, mode='lines', name='Epsilon',
                             line=dict(color='green', width=2)), row=2, col=1)

    # Comfort and CBT
    fig.add_trace(go.Scatter(x=episodes, y=comforts, mode='lines', name='Avg Comfort',
                             line=dict(color='purple', width=2)), row=2, col=2)
    fig.add_trace(go.Scatter(x=episodes, y=cbts, mode='lines', name='Avg CBT',
                             line=dict(color='orange', width=2), yaxis='y2'), row=2, col=2)

    fig.update_layout(height=500, showlegend=False)
    return fig


def create_episode_visualization(episode_data: dict) -> go.Figure:
    """Visualize a single episode."""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Environment State', 'Actions & Rewards', 'Q-Values Heatmap'),
        vertical_spacing=0.1,
        shared_xaxes=True
    )

    steps = list(range(len(episode_data['thi'])))

    # Environment state
    fig.add_trace(go.Scatter(x=steps, y=episode_data['thi'], mode='lines', name='THI',
                             line=dict(color='orange', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=steps, y=[c * 80 + 10 for c in episode_data['cbt']],
                             mode='lines', name='CBT (scaled)', line=dict(color='red', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=steps, y=episode_data['milk'], mode='lines', name='Milk',
                             line=dict(color='blue', width=2)), row=1, col=1)

    # Rewards
    fig.add_trace(go.Scatter(x=steps, y=episode_data['rewards'], mode='lines+markers',
                             name='Reward', line=dict(color='green', width=2)), row=2, col=1)

    # Action bars
    action_colors = ['gray', 'lightblue', 'blue', 'cyan']
    for i, action in enumerate(episode_data['actions']):
        fig.add_trace(go.Bar(x=[i], y=[0.5], marker_color=action_colors[action],
                             showlegend=False, opacity=0.6), row=2, col=1)

    # Q-Values heatmap
    if episode_data['q_values']:
        q_array = np.array(episode_data['q_values'])
        fig.add_trace(go.Heatmap(z=q_array.T, x=steps, y=['None', 'Fans', 'Fans+Spray', 'Spray'],
                                 colorscale='RdBu', name='Q-Values'), row=3, col=1)

    fig.update_layout(height=700, showlegend=True)
    return fig


def create_data_overview(data: pd.DataFrame) -> go.Figure:
    """Create overview visualization of the dataset."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('THI Distribution', 'CBT Distribution', 'THI vs CBT', 'Hourly Patterns')
    )

    # THI distribution
    fig.add_trace(go.Histogram(x=data['thi'], nbinsx=30, name='THI',
                               marker_color='orange'), row=1, col=1)

    # CBT distribution
    fig.add_trace(go.Histogram(x=data['cbt'], nbinsx=30, name='CBT',
                               marker_color='red'), row=1, col=2)

    # THI vs CBT scatter
    sample = data.sample(min(1000, len(data)))
    fig.add_trace(go.Scatter(x=sample['thi'], y=sample['cbt'], mode='markers',
                             marker=dict(size=4, opacity=0.5, color='blue'), name='THI vs CBT'), row=2, col=1)

    # Hourly patterns
    if 'hour' in data.columns:
        hourly_thi = data.groupby('hour')['thi'].mean()
        hourly_cbt = data.groupby('hour')['cbt'].mean()
        fig.add_trace(go.Scatter(x=hourly_thi.index, y=hourly_thi.values, mode='lines+markers',
                                 name='Avg THI', line=dict(color='orange')), row=2, col=2)
        fig.add_trace(go.Scatter(x=hourly_cbt.index, y=hourly_cbt.values * 2,
                                 mode='lines+markers', name='Avg CBT (scaled)', line=dict(color='red')), row=2, col=2)

    fig.update_layout(height=500, showlegend=True)
    return fig


# =============================================================================
# STREAMLIT APPLICATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="DQN Heat Stress Management",
        page_icon="🐄",
        layout="wide"
    )

    st.title("🐄 Deep Q-Learning for Heat Stress Management")
    st.markdown("*Train RL agents on real MmCows data or synthetic environments*")

    # Initialize data loader for real MmCows data
    data_loader = MmCowsDataLoader("data/mmcows_real/sensor_data")

    # Sidebar
    st.sidebar.header("⚙️ Configuration")

    # Data source selection
    st.sidebar.subheader("Data Source")
    data_exists = data_loader.check_data_exists()

    if data_exists:
        use_real_data = st.sidebar.checkbox("Use Real MmCows Data", value=True)
        if use_real_data:
            with st.spinner("Loading MmCows data..."):
                real_data = data_loader.load()
            if real_data is not None:
                st.sidebar.success(f"Loaded {len(real_data)} samples")
            else:
                st.sidebar.warning("Failed to load data, using synthetic")
                use_real_data = False
                real_data = None
        else:
            real_data = None
    else:
        st.sidebar.warning("MmCows data not found!")
        st.sidebar.code("python download_mmcows.py --synthetic")
        use_real_data = False
        real_data = None

    # Network architecture
    st.sidebar.subheader("Network Architecture")
    use_double_dqn = st.sidebar.checkbox("Double DQN", value=True)
    use_dueling = st.sidebar.checkbox("Dueling DQN", value=False)
    use_per = st.sidebar.checkbox("Prioritized Replay", value=False)
    use_noisy = st.sidebar.checkbox("Noisy Networks", value=False)

    # Hyperparameters
    st.sidebar.subheader("Hyperparameters")
    learning_rate = st.sidebar.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3], value=1e-3)
    gamma = st.sidebar.slider("Gamma (γ)", 0.9, 0.999, 0.99, 0.01)
    epsilon_decay = st.sidebar.slider("Epsilon Decay", 0.99, 0.999, 0.995, 0.001)
    batch_size = st.sidebar.select_slider("Batch Size", [32, 64, 128, 256], value=64)
    hidden_size = st.sidebar.select_slider("Hidden Size", [64, 128, 256], value=128)

    # Training settings
    st.sidebar.subheader("Training")
    n_episodes = st.sidebar.number_input("Episodes", 10, 500, 100)
    episode_length = st.sidebar.slider("Episode Length", 24, 96, 48)

    # Session state
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'metrics_history' not in st.session_state:
        st.session_state.metrics_history = []
    if 'last_episode_data' not in st.session_state:
        st.session_state.last_episode_data = None

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🎮 Training", "📊 Data Explorer", "📈 Results", "📚 Info"])

    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Training Controls")
            btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

            with btn_col1:
                start_training = st.button("🚀 Start", type="primary", use_container_width=True)

            with btn_col2:
                reset = st.button("🔄 Reset", use_container_width=True)

            with btn_col3:
                evaluate = st.button("📊 Evaluate", use_container_width=True)

            with btn_col4:
                save_model = st.button("💾 Save", use_container_width=True)

            if reset:
                st.session_state.agent = None
                st.session_state.metrics_history = []
                st.session_state.last_episode_data = None
                st.rerun()

            if start_training:
                config = DQNConfig(
                    state_dim=7,
                    action_dim=4,
                    hidden_sizes=[hidden_size, hidden_size],
                    learning_rate=learning_rate,
                    gamma=gamma,
                    epsilon_decay=epsilon_decay,
                    batch_size=batch_size,
                    use_double_dqn=use_double_dqn,
                    use_dueling=use_dueling,
                    use_per=use_per,
                    use_noisy=use_noisy
                )

                st.session_state.agent = DQNAgent(config)
                st.session_state.metrics_history = []

                env = HeatStressEnv(
                    data=real_data if use_real_data else None,
                    episode_length=episode_length
                )

                progress_bar = st.progress(0)
                status = st.empty()
                plot_area = st.empty()

                for episode in range(n_episodes):
                    metrics = train_episode(st.session_state.agent, env, episode)
                    st.session_state.metrics_history.append(metrics)
                    st.session_state.last_episode_data = metrics['episode_data']

                    progress_bar.progress((episode + 1) / n_episodes)
                    status.text(
                        f"Episode {episode + 1}/{n_episodes} | "
                        f"Reward: {metrics['total_reward']:.2f} | "
                        f"Loss: {metrics['avg_loss']:.4f} | "
                        f"ε: {metrics['epsilon']:.3f}"
                    )

                    if (episode + 1) % 5 == 0:
                        fig = create_training_plots(st.session_state.metrics_history)
                        plot_area.plotly_chart(fig, use_container_width=True)

                st.success(f"Training complete! Final reward: {metrics['total_reward']:.2f}")

            if evaluate and st.session_state.agent:
                env = HeatStressEnv(
                    data=real_data if use_real_data else None,
                    episode_length=episode_length
                )
                rewards = []
                for _ in range(10):
                    state, _ = env.reset()
                    ep_reward = 0
                    while True:
                        action = st.session_state.agent.select_action(state, training=False)
                        state, reward, term, trunc, _ = env.step(action)
                        ep_reward += reward
                        if term or trunc:
                            break
                    rewards.append(ep_reward)
                st.info(f"Evaluation: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")

            if save_model and st.session_state.agent:
                os.makedirs("models", exist_ok=True)
                path = f"models/dqn_heat_stress_{int(time.time())}.pt"
                st.session_state.agent.save(path)
                st.success(f"Model saved to {path}")

        with col2:
            st.subheader("Agent Status")
            if st.session_state.agent:
                cfg = st.session_state.agent.config
                st.markdown(f"""
                **Architecture:**
                - Double DQN: {'✅' if cfg.use_double_dqn else '❌'}
                - Dueling: {'✅' if cfg.use_dueling else '❌'}
                - PER: {'✅' if cfg.use_per else '❌'}
                - Noisy: {'✅' if cfg.use_noisy else '❌'}

                **Training:**
                - Epsilon: {st.session_state.agent.epsilon:.4f}
                - Buffer: {len(st.session_state.agent.memory)}
                - Steps: {st.session_state.agent.steps_done}
                """)
            else:
                st.info("No agent. Click Start to begin.")

    with tab2:
        st.subheader("Dataset Explorer")

        if use_real_data and real_data is not None:
            stats = data_loader.get_statistics()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Samples", f"{stats['n_samples']:,}")
            col2.metric("Cows", stats['n_cows'])
            col3.metric("Heat Stress %", f"{stats['heat_stress_ratio']*100:.1f}%")
            col4.metric("THI Range", f"{stats['thi_range'][0]:.0f}-{stats['thi_range'][1]:.0f}")

            fig = create_data_overview(real_data)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Raw Data Sample"):
                st.dataframe(real_data.head(100))
        else:
            st.info("Load real MmCows data to explore. Run:")
            st.code("python download_mmcows.py --synthetic")

    with tab3:
        st.subheader("Training Results")

        if st.session_state.metrics_history:
            fig = create_training_plots(st.session_state.metrics_history)
            st.plotly_chart(fig, use_container_width=True)

            if st.session_state.last_episode_data:
                st.subheader("Last Episode Details")
                fig = create_episode_visualization(st.session_state.last_episode_data)
                st.plotly_chart(fig, use_container_width=True)

            # Statistics
            df = pd.DataFrame([{
                'Episode': m['episode'],
                'Reward': m['total_reward'],
                'Loss': m['avg_loss'],
                'Epsilon': m['epsilon'],
                'Comfort': m.get('avg_comfort', 0),
                'CBT': m.get('avg_cbt', 38.5)
            } for m in st.session_state.metrics_history])

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Best Reward", f"{df['Reward'].max():.2f}")
            col2.metric("Avg (last 10)", f"{df['Reward'].tail(10).mean():.2f}")
            col3.metric("Avg Comfort", f"{df['Comfort'].tail(10).mean():.2f}")
            col4.metric("Avg CBT", f"{df['CBT'].tail(10).mean():.2f}°C")
        else:
            st.info("Train an agent to see results.")

    with tab4:
        st.subheader("About")
        st.markdown("""
        ### Deep Q-Learning for Heat Stress Management

        This application trains DQN agents to manage heat stress in dairy cattle
        using the MmCows dataset.

        **State Space (7 features):**
        - THI (Temperature-Humidity Index)
        - CBT (Core Body Temperature)
        - Hour (sin/cos encoded)
        - Lying ratio
        - Activity level
        - Previous milk yield

        **Actions:**
        - 0: No intervention
        - 1: Fans only
        - 2: Fans + sprinklers
        - 3: Water spray

        **Reward:**
        - Milk production maintenance
        - Comfort score
        - Energy penalty

        **DQN Enhancements:**
        - **Double DQN**: Reduces Q-value overestimation
        - **Dueling DQN**: Separates value and advantage
        - **PER**: Prioritized Experience Replay
        - **Noisy Networks**: Parametric exploration
        """)


if __name__ == "__main__":
    main()
