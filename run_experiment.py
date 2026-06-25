#!/usr/bin/env python3
"""
Run DQN Experiment on Real MmCows Data
======================================

Trains multiple DQN variants and generates results for the paper.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path
import json
from datetime import datetime

# Import from streamlit_dqn
import sys
sys.path.insert(0, '.')

from streamlit_dqn import (
    MmCowsDataLoader, HeatStressEnv, DQNAgent, DQNConfig,
    train_episode, ReplayBuffer
)


def run_baseline(env, baseline_type: str, n_episodes: int = 50):
    """Run baseline policy."""
    all_rewards = []
    all_comfort = []
    all_cbt = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_comfort = []
        episode_cbt = []

        while True:
            thi = state[0]
            cbt = state[1]

            if baseline_type == "no_action":
                action = 0
            elif baseline_type == "thi_threshold":
                action = 2 if thi > 72 else 0
            elif baseline_type == "cbt_threshold":
                action = 2 if cbt > 39.0 else 0
            elif baseline_type == "random":
                action = np.random.randint(0, 4)
            else:
                action = 0

            state, reward, term, trunc, info = env.step(action)
            episode_reward += reward
            episode_comfort.append(info.get('comfort', 0.5))
            episode_cbt.append(info.get('cbt', 38.5))

            if term or trunc:
                break

        all_rewards.append(episode_reward)
        all_comfort.append(np.mean(episode_comfort))
        all_cbt.append(np.mean(episode_cbt))

    return {
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_comfort': np.mean(all_comfort),
        'mean_cbt': np.mean(all_cbt)
    }


def train_agent(env, config: DQNConfig, n_episodes: int = 200, name: str = "DQN"):
    """Train a DQN agent and return metrics."""
    agent = DQNAgent(config)

    metrics = {
        'episodes': [],
        'rewards': [],
        'losses': [],
        'epsilons': [],
        'comfort': [],
        'cbt': []
    }

    print(f"\nTraining {name}...")

    for episode in range(n_episodes):
        result = train_episode(agent, env, episode)

        metrics['episodes'].append(episode)
        metrics['rewards'].append(result['total_reward'])
        metrics['losses'].append(result['avg_loss'])
        metrics['epsilons'].append(result['epsilon'])
        metrics['comfort'].append(result.get('avg_comfort', 0.5))
        metrics['cbt'].append(result.get('avg_cbt', 38.5))

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(metrics['rewards'][-50:])
            print(f"  Episode {episode + 1}/{n_episodes}, Avg Reward: {avg_reward:.3f}")

    return agent, metrics


def evaluate_agent(agent, env, n_episodes: int = 50):
    """Evaluate trained agent."""
    rewards = []
    comfort_scores = []
    cbt_values = []
    actions_taken = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_comfort = []
        episode_cbt = []
        episode_actions = []

        while True:
            action = agent.select_action(state, training=False)
            state, reward, term, trunc, info = env.step(action)

            episode_reward += reward
            episode_comfort.append(info.get('comfort', 0.5))
            episode_cbt.append(info.get('cbt', 38.5))
            episode_actions.append(action)

            if term or trunc:
                break

        rewards.append(episode_reward)
        comfort_scores.append(np.mean(episode_comfort))
        cbt_values.append(np.mean(episode_cbt))
        actions_taken.extend(episode_actions)

    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_comfort': np.mean(comfort_scores),
        'mean_cbt': np.mean(cbt_values),
        'action_distribution': np.bincount(actions_taken, minlength=4) / len(actions_taken)
    }


def plot_training_curves(all_metrics: dict, output_dir: Path):
    """Plot training curves for all agents."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Rewards
    ax = axes[0, 0]
    for i, (name, metrics) in enumerate(all_metrics.items()):
        rewards = pd.Series(metrics['rewards']).rolling(window=20, min_periods=1).mean()
        ax.plot(metrics['episodes'], rewards, label=name, color=colors[i % len(colors)], linewidth=2)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Average Reward', fontsize=12)
    ax.set_title('Training Rewards', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Loss
    ax = axes[0, 1]
    for i, (name, metrics) in enumerate(all_metrics.items()):
        losses = pd.Series(metrics['losses']).rolling(window=20, min_periods=1).mean()
        ax.plot(metrics['episodes'], losses, label=name, color=colors[i % len(colors)], linewidth=2)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Comfort
    ax = axes[1, 0]
    for i, (name, metrics) in enumerate(all_metrics.items()):
        comfort = pd.Series(metrics['comfort']).rolling(window=20, min_periods=1).mean()
        ax.plot(metrics['episodes'], comfort, label=name, color=colors[i % len(colors)], linewidth=2)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Comfort Score', fontsize=12)
    ax.set_title('Average Comfort During Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # CBT
    ax = axes[1, 1]
    for i, (name, metrics) in enumerate(all_metrics.items()):
        cbt = pd.Series(metrics['cbt']).rolling(window=20, min_periods=1).mean()
        ax.plot(metrics['episodes'], cbt, label=name, color=colors[i % len(colors)], linewidth=2)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Core Body Temperature (°C)', fontsize=12)
    ax.set_title('Average CBT During Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'training_curves.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved training curves to {output_dir}")


def plot_comparison(results: dict, output_dir: Path):
    """Plot comparison bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    methods = list(results.keys())
    rewards = [results[m]['mean_reward'] for m in methods]
    reward_stds = [results[m]['std_reward'] for m in methods]
    comfort = [results[m]['mean_comfort'] for m in methods]
    cbt = [results[m]['mean_cbt'] for m in methods]

    x = np.arange(len(methods))

    # Rewards
    ax = axes[0]
    colors = ['#ff6b6b' if 'Baseline' in m or m in ['No Action', 'THI Threshold', 'CBT Threshold', 'Random']
              else '#4ecdc4' for m in methods]
    bars = ax.bar(x, rewards, yerr=reward_stds, capsize=5, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('Performance Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Comfort
    ax = axes[1]
    bars = ax.bar(x, comfort, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Comfort Score', fontsize=12)
    ax.set_title('Comfort Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # CBT
    ax = axes[2]
    bars = ax.bar(x, cbt, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Mean CBT (°C)', fontsize=12)
    ax.set_title('CBT Comparison', fontsize=14, fontweight='bold')
    ax.axhline(y=39.0, color='red', linestyle='--', label='Heat stress threshold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'comparison.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved comparison plot to {output_dir}")


def plot_action_distribution(results: dict, output_dir: Path):
    """Plot action distribution for RL agents."""
    fig, ax = plt.subplots(figsize=(10, 6))

    action_names = ['No Action', 'Fans', 'Fans+Spray', 'Water Spray']
    rl_methods = [m for m in results.keys() if 'action_distribution' in results[m]]

    x = np.arange(len(action_names))
    width = 0.8 / len(rl_methods)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, method in enumerate(rl_methods):
        dist = results[method]['action_distribution']
        offset = (i - len(rl_methods)/2 + 0.5) * width
        bars = ax.bar(x + offset, dist * 100, width, label=method, color=colors[i % len(colors)])

    ax.set_xticks(x)
    ax.set_xticklabels(action_names, fontsize=11)
    ax.set_ylabel('Action Frequency (%)', fontsize=12)
    ax.set_title('Action Distribution by RL Agent', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'action_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'action_distribution.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved action distribution to {output_dir}")


def plot_data_overview(data: pd.DataFrame, output_dir: Path):
    """Plot data overview."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # THI distribution
    ax = axes[0, 0]
    ax.hist(data['thi'], bins=30, color='#ff7f0e', edgecolor='black', alpha=0.7)
    ax.axvline(x=72, color='red', linestyle='--', linewidth=2, label='Heat stress threshold (THI=72)')
    ax.set_xlabel('Temperature-Humidity Index (THI)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('THI Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)

    # CBT distribution
    ax = axes[0, 1]
    ax.hist(data['cbt'], bins=30, color='#d62728', edgecolor='black', alpha=0.7)
    ax.axvline(x=39.0, color='darkred', linestyle='--', linewidth=2, label='Fever threshold (39°C)')
    ax.set_xlabel('Core Body Temperature (°C)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('CBT Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)

    # THI vs CBT
    ax = axes[1, 0]
    scatter = ax.scatter(data['thi'], data['cbt'], c=data['hour'], cmap='twilight', alpha=0.5, s=10)
    ax.set_xlabel('THI', fontsize=12)
    ax.set_ylabel('CBT (°C)', fontsize=12)
    ax.set_title('THI vs CBT Relationship', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Hour of Day')

    # Hourly patterns
    ax = axes[1, 1]
    hourly_thi = data.groupby('hour')['thi'].mean()
    hourly_cbt = data.groupby('hour')['cbt'].mean()

    ax2 = ax.twinx()
    line1, = ax.plot(hourly_thi.index, hourly_thi.values, 'o-', color='#ff7f0e', linewidth=2, markersize=6, label='THI')
    line2, = ax2.plot(hourly_cbt.index, hourly_cbt.values, 's-', color='#d62728', linewidth=2, markersize=6, label='CBT')

    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('THI', color='#ff7f0e', fontsize=12)
    ax2.set_ylabel('CBT (°C)', color='#d62728', fontsize=12)
    ax.set_title('Diurnal Patterns', fontsize=14, fontweight='bold')
    ax.legend([line1, line2], ['THI', 'CBT'], loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'data_overview.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'data_overview.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved data overview to {output_dir}")


def main():
    # Setup
    output_dir = Path("results/paper_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DQN EXPERIMENT FOR HEAT STRESS MANAGEMENT")
    print("=" * 60)

    # Load real data
    print("\nLoading real MmCows data...")
    loader = MmCowsDataLoader("data/mmcows_real/sensor_data")
    data = loader.load()

    if data is None:
        print("Error: Could not load data!")
        return

    stats = loader.get_statistics()
    print(f"  Samples: {stats['n_samples']}")
    print(f"  Cows: {stats['n_cows']}")
    print(f"  THI range: {stats['thi_range'][0]:.1f} - {stats['thi_range'][1]:.1f}")
    print(f"  Heat stress ratio: {stats['heat_stress_ratio']*100:.1f}%")

    # Plot data overview
    plot_data_overview(data, output_dir)

    # Create environment
    env = HeatStressEnv(data=data, episode_length=48)

    # Training parameters
    n_episodes = 200

    # Define agent configurations
    configs = {
        'Standard DQN': DQNConfig(
            state_dim=7, action_dim=4,
            use_double_dqn=False, use_dueling=False, use_per=False, use_noisy=False
        ),
        'Double DQN': DQNConfig(
            state_dim=7, action_dim=4,
            use_double_dqn=True, use_dueling=False, use_per=False, use_noisy=False
        ),
        'Dueling DQN': DQNConfig(
            state_dim=7, action_dim=4,
            use_double_dqn=True, use_dueling=True, use_per=False, use_noisy=False
        ),
        'Double Dueling DQN + PER': DQNConfig(
            state_dim=7, action_dim=4,
            use_double_dqn=True, use_dueling=True, use_per=True, use_noisy=False
        )
    }

    # Train all agents
    all_training_metrics = {}
    trained_agents = {}

    for name, config in configs.items():
        agent, metrics = train_agent(env, config, n_episodes=n_episodes, name=name)
        all_training_metrics[name] = metrics
        trained_agents[name] = agent

    # Plot training curves
    plot_training_curves(all_training_metrics, output_dir)

    # Evaluate baselines
    print("\nEvaluating baselines...")
    results = {}

    for baseline in ['no_action', 'thi_threshold', 'cbt_threshold', 'random']:
        baseline_name = baseline.replace('_', ' ').title()
        results[baseline_name] = run_baseline(env, baseline, n_episodes=50)
        print(f"  {baseline_name}: Reward = {results[baseline_name]['mean_reward']:.3f}")

    # Evaluate trained agents
    print("\nEvaluating trained agents...")
    for name, agent in trained_agents.items():
        eval_results = evaluate_agent(agent, env, n_episodes=50)
        results[name] = eval_results
        print(f"  {name}: Reward = {eval_results['mean_reward']:.3f} ± {eval_results['std_reward']:.3f}")

    # Plot comparisons
    plot_comparison(results, output_dir)
    plot_action_distribution(results, output_dir)

    # Save results to JSON
    results_json = {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else
                        (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                        for kk, vv in v.items()}
                    for k, v in results.items()}

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_json, f, indent=2)

    # Print final results table
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"{'Method':<30} {'Reward':>10} {'Comfort':>10} {'CBT':>10}")
    print("-" * 60)
    for name, res in results.items():
        print(f"{name:<30} {res['mean_reward']:>10.3f} {res['mean_comfort']:>10.3f} {res['mean_cbt']:>10.2f}")

    print(f"\nResults saved to {output_dir}")

    return results, stats


if __name__ == "__main__":
    results, stats = main()
