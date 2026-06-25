#!/usr/bin/env python3
"""
Paper A: Deep Reinforcement Learning Agent for Heat Stress Management
======================================================================

Main entry point for training and evaluating RL agents for autonomous
heat stress intervention in dairy cattle.

Usage:
    python main.py --mode train --agent ppo --episodes 1000
    python main.py --mode evaluate --model models/best_model.zip
    python main.py --mode baseline --baseline thi_threshold

Authors:
    Jan Saro, Ing., Ph.D., MBA
        Department of Systems Engineering, Faculty of Economics and Management,
        Czech University of Life Sciences Prague, Prague, Czech Republic
    Jana Mazancová, Ph.D.
        Department of Sustainable Technologies,
        Czech University of Life Sciences Prague, Prague, Czech Republic
    Jaromír Ducháček, Ing., Ph.D.
        Department of Animal Science,
        Czech University of Life Sciences Prague, Prague, Czech Republic

Journal: Livestock Science (IF 2.12, Q2)
"""

import argparse
import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# RL libraries
import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.environment.heat_stress_env import HeatStressEnv
from src.data.mmcows_loader import MmCowsLoader


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_environment(config: dict, data: pd.DataFrame = None) -> gym.Env:
    """Create and wrap the Heat Stress environment."""
    reward_weights = {
        "milk": config['environment']['reward']['milk_weight'],
        "comfort": config['environment']['reward']['comfort_weight'],
        "energy": config['environment']['reward']['energy_penalty']
    }
    
    env = HeatStressEnv(
        data=data,
        reward_weights=reward_weights,
        episode_length=48
    )
    
    # Wrap with Monitor for logging
    log_dir = "results/logs/"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)
    
    return env


def create_agent(agent_type: str, env: gym.Env, config: dict):
    """Create RL agent based on type."""
    agent_config = config['agent']
    
    common_params = {
        'env': env,
        'verbose': 1,
        'gamma': agent_config['gamma'],
        'learning_rate': agent_config['learning_rate'],
        'tensorboard_log': "results/tensorboard/",
        'seed': config['experiment']['seed']
    }
    
    if agent_type.lower() == 'dqn':
        agent = DQN(
            'MlpPolicy',
            **common_params,
            buffer_size=agent_config['buffer_size'],
            batch_size=agent_config['batch_size'],
            exploration_fraction=agent_config['dqn']['exploration_fraction'],
            exploration_final_eps=agent_config['dqn']['exploration_final_eps'],
            target_update_interval=agent_config['dqn']['target_update_interval']
        )
    elif agent_type.lower() == 'ppo':
        agent = PPO(
            'MlpPolicy',
            **common_params,
            n_steps=agent_config['ppo']['n_steps'],
            n_epochs=agent_config['ppo']['n_epochs'],
            clip_range=agent_config['ppo']['clip_range'],
            ent_coef=agent_config['ppo']['ent_coef'],
            vf_coef=agent_config['ppo']['vf_coef'],
            batch_size=agent_config['batch_size']
        )
    elif agent_type.lower() == 'a2c':
        agent = A2C(
            'MlpPolicy',
            **common_params,
            n_steps=5
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    return agent


def run_baseline(env: gym.Env, baseline_type: str, n_episodes: int = 100) -> dict:
    """Run baseline policy evaluation."""
    print(f"\nRunning baseline: {baseline_type}")
    
    all_rewards = []
    all_milk = []
    all_energy = []
    all_comfort = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_milk = 0
        episode_energy = 0
        episode_comfort = 0
        steps = 0
        
        while True:
            # Select action based on baseline policy
            thi, cbt = obs[0], obs[1]
            
            if baseline_type == "no_action":
                action = 0
            elif baseline_type == "thi_threshold":
                action = 2 if thi > 72 else 0  # Intensive cooling if THI > 72
            elif baseline_type == "cbt_threshold":
                action = 2 if cbt > 39.5 else 0  # Intensive cooling if CBT > 39.5
            elif baseline_type == "combined":
                if thi > 72 and cbt > 39.0:
                    action = 2
                elif thi > 68 or cbt > 38.8:
                    action = 1
                else:
                    action = 0
            elif baseline_type == "random":
                action = env.action_space.sample()
            else:
                action = 0
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_milk += info.get('milk_change', 0)
            episode_energy += info.get('energy_cost', 0)
            episode_comfort += info.get('comfort', 0)
            steps += 1
            
            if terminated or truncated:
                break
        
        all_rewards.append(episode_reward)
        all_milk.append(episode_milk)
        all_energy.append(episode_energy)
        all_comfort.append(episode_comfort / steps if steps > 0 else 0)
    
    results = {
        'baseline': baseline_type,
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_milk_change': np.mean(all_milk),
        'mean_energy': np.mean(all_energy),
        'mean_comfort': np.mean(all_comfort)
    }
    
    print(f"  Mean reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"  Mean milk change: {results['mean_milk_change']:.3f}")
    print(f"  Mean energy cost: {results['mean_energy']:.3f}")
    print(f"  Mean comfort: {results['mean_comfort']:.3f}")
    
    return results


def train(args, config: dict):
    """Train RL agent."""
    print("\n" + "=" * 60)
    print("TRAINING RL AGENT FOR HEAT STRESS MANAGEMENT")
    print("=" * 60)
    
    # Load data
    if args.use_real_data:
        loader = MmCowsLoader(config['data']['mmcows_path'])
        data = loader.load_for_rl()
    else:
        data = None
        print("Using synthetic data for training")
    
    # Create environment
    env = create_environment(config, data)
    eval_env = create_environment(config, data)
    
    # Create agent
    agent = create_agent(args.agent, env, config)
    
    # Callbacks
    os.makedirs("models/", exist_ok=True)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/",
        log_path="results/logs/",
        eval_freq=config['training']['eval_freq'],
        n_eval_episodes=config['training']['n_eval_episodes'],
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config['training']['save_freq'],
        save_path="models/checkpoints/",
        name_prefix=f"heat_stress_{args.agent}"
    )
    
    # Train
    print(f"\nTraining {args.agent.upper()} agent for {config['training']['total_timesteps']} timesteps...")
    
    agent.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=[eval_callback, checkpoint_callback],
        log_interval=config['training']['log_interval'],
        progress_bar=True
    )
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/heat_stress_{args.agent}_{timestamp}.zip"
    agent.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Final evaluation
    print("\nFinal evaluation...")
    mean_reward, std_reward = evaluate_policy(agent, eval_env, n_eval_episodes=50)
    print(f"Mean reward: {mean_reward:.3f} ± {std_reward:.3f}")
    
    return agent


def evaluate(args, config: dict):
    """Evaluate trained model."""
    print("\n" + "=" * 60)
    print("EVALUATING RL AGENT")
    print("=" * 60)
    
    # Load data
    if args.use_real_data:
        loader = MmCowsLoader(config['data']['mmcows_path'])
        data = loader.load_for_rl()
    else:
        data = None
    
    # Create environment
    env = create_environment(config, data)
    
    # Load model
    model_path = args.model
    if model_path.endswith('.zip'):
        # Detect agent type from filename
        if 'dqn' in model_path.lower():
            agent = DQN.load(model_path, env=env)
        elif 'ppo' in model_path.lower():
            agent = PPO.load(model_path, env=env)
        else:
            agent = PPO.load(model_path, env=env)  # Default to PPO
    else:
        raise ValueError(f"Invalid model path: {model_path}")
    
    print(f"Loaded model from: {model_path}")
    
    # Evaluate
    n_episodes = config['evaluation']['n_episodes']
    mean_reward, std_reward = evaluate_policy(agent, env, n_eval_episodes=n_episodes)
    
    print(f"\nEvaluation Results ({n_episodes} episodes):")
    print(f"  Mean reward: {mean_reward:.3f} ± {std_reward:.3f}")
    
    # Compare with baselines
    print("\n" + "-" * 40)
    print("BASELINE COMPARISONS")
    print("-" * 40)
    
    baselines = ["no_action", "thi_threshold", "cbt_threshold", "combined", "random"]
    baseline_results = []
    
    for baseline in baselines:
        result = run_baseline(env, baseline, n_episodes=n_episodes)
        baseline_results.append(result)
    
    # Add RL agent result
    baseline_results.append({
        'baseline': f'RL ({args.agent})',
        'mean_reward': mean_reward,
        'std_reward': std_reward
    })
    
    # Save results
    results_df = pd.DataFrame(baseline_results)
    results_path = f"results/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    return results_df


def run_all_baselines(args, config: dict):
    """Run all baseline evaluations."""
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION")
    print("=" * 60)
    
    # Create environment
    env = create_environment(config, data=None)
    
    baselines = ["no_action", "thi_threshold", "cbt_threshold", "combined", "random"]
    results = []
    
    for baseline in baselines:
        result = run_baseline(env, baseline, n_episodes=args.episodes)
        results.append(result)
    
    # Save results
    results_df = pd.DataFrame(results)
    print("\n" + "-" * 40)
    print("SUMMARY")
    print("-" * 40)
    print(results_df.to_string(index=False))
    
    results_path = f"results/baselines_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="RL Agent for Dairy Cattle Heat Stress Management"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "baseline"],
        default="train",
        help="Mode: train, evaluate, or baseline"
    )
    
    parser.add_argument(
        "--agent",
        type=str,
        choices=["dqn", "ppo", "a2c"],
        default="ppo",
        help="RL agent type"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_model.zip",
        help="Path to trained model (for evaluate mode)"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes"
    )
    
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="Use real MmCows data (must be downloaded)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create directories
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/tensorboard", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    
    # Run
    if args.mode == "train":
        train(args, config)
    elif args.mode == "evaluate":
        evaluate(args, config)
    elif args.mode == "baseline":
        run_all_baselines(args, config)


if __name__ == "__main__":
    main()
