#!/usr/bin/env python3
"""
Comprehensive experiments for Heat Stress RL paper.
Includes multiple algorithms, ablation study, and statistical analysis.
"""

import os
import sys
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).parent))

from stable_baselines3 import PPO, DQN, A2C, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment.heat_stress_env_v3 import HeatStressEnvV3


def run_baseline(env, baseline_type: str, n_episodes: int = 100) -> list:
    """Run baseline and return all episode rewards."""
    all_rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0

        while True:
            thi, cbt, _, _, hour, _, heat_load, _, sens, fatigue, thi_trend = obs

            if baseline_type == "no_action":
                action = 0
            elif baseline_type == "thi_threshold":
                action = 2 if thi > 72 else (1 if thi > 68 else 0)
            elif baseline_type == "cbt_threshold":
                action = 2 if cbt > 39.5 else (1 if cbt > 39.0 else 0)
            elif baseline_type == "combined":
                if thi > 72 and cbt > 39.0:
                    action = 2
                elif thi > 68 or cbt > 38.8:
                    action = 1
                else:
                    action = 0
            elif baseline_type == "proactive":
                if cbt > 39.5 or thi > 75:
                    action = 2
                elif cbt > 39.0 or thi > 70:
                    action = 1
                elif thi_trend > 0.3 and thi > 65:
                    action = 1
                else:
                    action = 0
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        all_rewards.append(episode_reward)

    return all_rewards


def evaluate_agent(agent, env, n_episodes: int = 100) -> list:
    """Evaluate agent and return all episode rewards."""
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        while True:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, trunc, _ = env.step(int(action))
            ep_reward += reward
            if done or trunc:
                break
        rewards.append(ep_reward)
    return rewards


def train_algorithm(algo_name: str, timesteps: int = 300000, seed: int = 42):
    """Train a specific algorithm."""
    os.makedirs(f"models_exp/{algo_name}", exist_ok=True)
    os.makedirs(f"results/logs_{algo_name}", exist_ok=True)

    env = HeatStressEnvV3()
    env = Monitor(env, f"results/logs_{algo_name}/train")

    eval_env = HeatStressEnvV3()
    eval_env = Monitor(eval_env, f"results/logs_{algo_name}/eval")

    common_kwargs = {
        "policy": "MlpPolicy",
        "env": env,
        "verbose": 1,
        "seed": seed,
        "tensorboard_log": f"results/tensorboard_{algo_name}/",
    }

    if algo_name == "PPO":
        agent = PPO(
            **common_kwargs,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            policy_kwargs=dict(net_arch=[256, 256]),
        )
    elif algo_name == "DQN":
        agent = DQN(
            **common_kwargs,
            learning_rate=5e-4,
            buffer_size=100000,
            learning_starts=5000,
            batch_size=128,
            gamma=0.99,
            exploration_fraction=0.3,
            exploration_final_eps=0.05,
            policy_kwargs=dict(net_arch=[256, 256]),
        )
    elif algo_name == "A2C":
        agent = A2C(
            **common_kwargs,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[256, 256]),
        )
    elif algo_name == "SAC":
        # SAC needs continuous action space, so we wrap discrete
        from gymnasium.spaces import Box
        # Skip SAC for discrete action space
        print(f"SAC requires continuous actions, skipping...")
        return None
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"models_exp/{algo_name}/",
        eval_freq=10000,
        n_eval_episodes=20,
        deterministic=True
    )

    print(f"\nTraining {algo_name} ({timesteps} steps)...")
    agent.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=True)
    agent.save(f"models_exp/{algo_name}/final.zip")

    return agent


def run_multi_seed_experiment(algo_name: str, seeds: list = [42, 123, 456], timesteps: int = 200000):
    """Train algorithm with multiple seeds for statistical significance."""
    results = []

    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"Training {algo_name} with seed {seed}")
        print('='*50)

        agent = train_algorithm(algo_name, timesteps=timesteps, seed=seed)
        if agent is None:
            continue

        env = HeatStressEnvV3()
        rewards = evaluate_agent(agent, env, n_episodes=50)
        results.append({
            "seed": seed,
            "mean": np.mean(rewards),
            "std": np.std(rewards),
            "rewards": rewards
        })

    return results


def ablation_study():
    """
    Ablation study: Train PPO on environments with different features disabled.
    """
    print("\n" + "="*60)
    print("ABLATION STUDY")
    print("="*60)

    results = {}

    # Full V3 environment
    print("\n1. Full V3 (all features)...")
    agent_full = train_algorithm("PPO", timesteps=200000)
    env = HeatStressEnvV3()
    results["Full V3"] = evaluate_agent(agent_full, env)

    # We'll create modified environments for ablation
    # For simplicity, we'll test the existing model on modified conditions

    return results


def statistical_tests(results: dict):
    """Perform statistical significance tests."""
    print("\n" + "="*60)
    print("STATISTICAL TESTS")
    print("="*60)

    tests = {}

    # Get baseline (combined) as reference
    if "combined" in results and "PPO" in results:
        baseline_rewards = results["combined"]
        ppo_rewards = results["PPO"]

        # Welch's t-test
        t_stat, p_value = stats.ttest_ind(ppo_rewards, baseline_rewards, equal_var=False)

        # Mann-Whitney U test (non-parametric)
        u_stat, p_mw = stats.mannwhitneyu(ppo_rewards, baseline_rewards, alternative='greater')

        # Effect size (Cohen's d)
        cohens_d = (np.mean(ppo_rewards) - np.mean(baseline_rewards)) / np.sqrt(
            (np.std(ppo_rewards)**2 + np.std(baseline_rewards)**2) / 2
        )

        tests["PPO vs Combined"] = {
            "t_statistic": t_stat,
            "p_value_ttest": p_value,
            "u_statistic": u_stat,
            "p_value_mannwhitney": p_mw,
            "cohens_d": cohens_d,
            "significant_005": p_value < 0.05,
            "significant_001": p_value < 0.01
        }

        print(f"\nPPO vs Combined Baseline:")
        print(f"  Welch's t-test: t={t_stat:.3f}, p={p_value:.6f}")
        print(f"  Mann-Whitney U: U={u_stat:.1f}, p={p_mw:.6f}")
        print(f"  Cohen's d: {cohens_d:.3f}")
        print(f"  Significant (p<0.01): {p_value < 0.01}")

    return tests


def generate_visualizations(results: dict, save_dir: str = "results/figures"):
    """Generate publication-quality figures."""
    os.makedirs(save_dir, exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Bar plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(results.keys())
    means = [np.mean(results[m]) for m in methods]
    stds = [np.std(results[m]) for m in methods]

    colors = ['#2ecc71' if 'PPO' in m or 'DQN' in m or 'A2C' in m else '#3498db' for m in methods]

    bars = ax.bar(methods, means, yerr=stds, capsize=5, color=colors, edgecolor='black')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Mean Episode Reward', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title('Performance Comparison: RL Agents vs Baselines', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/comparison_bar.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/comparison_bar.pdf", bbox_inches='tight')
    plt.close()

    # 2. Box plot
    fig, ax = plt.subplots(figsize=(10, 6))

    data = [results[m] for m in methods]
    bp = ax.boxplot(data, labels=methods, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title('Reward Distribution by Method', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/comparison_box.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_dir}/comparison_box.pdf", bbox_inches='tight')
    plt.close()

    print(f"\nFigures saved to {save_dir}/")


def run_all_experiments():
    """Run comprehensive experiments."""
    os.makedirs("results/experiments", exist_ok=True)

    results = {}
    env = HeatStressEnvV3()

    # 1. Baselines
    print("\n" + "="*60)
    print("EVALUATING BASELINES")
    print("="*60)

    for baseline in ["no_action", "thi_threshold", "cbt_threshold", "combined", "proactive"]:
        print(f"\nEvaluating {baseline}...")
        rewards = run_baseline(env, baseline, n_episodes=100)
        results[baseline] = rewards
        print(f"  {baseline}: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")

    # 2. Train RL algorithms
    print("\n" + "="*60)
    print("TRAINING RL ALGORITHMS")
    print("="*60)

    for algo in ["PPO", "DQN", "A2C"]:
        print(f"\n--- {algo} ---")
        agent = train_algorithm(algo, timesteps=300000)
        if agent:
            rewards = evaluate_agent(agent, env, n_episodes=100)
            results[algo] = rewards
            print(f"  {algo}: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")

    # 3. Statistical tests
    stats_results = statistical_tests(results)

    # 4. Visualizations
    generate_visualizations(results)

    # 5. Save results
    summary = {
        method: {
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "min": float(np.min(rewards)),
            "max": float(np.max(rewards)),
            "median": float(np.median(rewards))
        }
        for method, rewards in results.items()
    }
    summary["statistical_tests"] = stats_results

    with open("results/experiments/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 6. Print final ranking
    print("\n" + "="*60)
    print("FINAL RANKING")
    print("="*60)

    ranking = sorted(
        [(m, np.mean(r), np.std(r)) for m, r in results.items()],
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\n{'Rank':<6}{'Method':<15}{'Mean':>10}{'Std':>10}{'Type':<12}")
    print("-" * 53)
    for i, (method, mean, std) in enumerate(ranking, 1):
        method_type = "RL Agent" if method in ["PPO", "DQN", "A2C"] else "Baseline"
        marker = " ***" if method in ["PPO", "DQN", "A2C"] else ""
        print(f"{i:<6}{method:<15}{mean:>10.2f}{std:>10.2f}  {method_type}{marker}")

    return results, summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "quick", "viz"], default="full")
    parser.add_argument("--timesteps", type=int, default=300000)
    args = parser.parse_args()

    if args.mode == "full":
        results, summary = run_all_experiments()
    elif args.mode == "quick":
        # Quick test with fewer timesteps
        print("Running quick experiment (100k steps)...")
        results = {}
        env = HeatStressEnvV3()

        for baseline in ["combined", "proactive"]:
            results[baseline] = run_baseline(env, baseline, n_episodes=50)

        agent = train_algorithm("PPO", timesteps=100000)
        results["PPO"] = evaluate_agent(agent, env, n_episodes=50)

        generate_visualizations(results)
    elif args.mode == "viz":
        # Just regenerate visualizations from existing models
        env = HeatStressEnvV3()
        results = {}

        for baseline in ["no_action", "thi_threshold", "cbt_threshold", "combined", "proactive"]:
            results[baseline] = run_baseline(env, baseline, n_episodes=100)

        for algo in ["PPO", "DQN", "A2C"]:
            model_path = f"models_exp/{algo}/best_model.zip"
            if os.path.exists(model_path):
                if algo == "PPO":
                    agent = PPO.load(model_path)
                elif algo == "DQN":
                    agent = DQN.load(model_path)
                elif algo == "A2C":
                    agent = A2C.load(model_path)
                results[algo] = evaluate_agent(agent, env, n_episodes=100)

        generate_visualizations(results)
        statistical_tests(results)
