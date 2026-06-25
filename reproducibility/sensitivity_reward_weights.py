#!/usr/bin/env python3
"""
Reward-weight sensitivity analysis (reviewer request).
======================================================

The reviewer asked for "a sensitivity analysis of the reward weights".
The reward is  r = alpha*milk + beta*comfort - gamma*energy (+bonus).
The published configuration is (alpha, beta, gamma) = (0.4, 0.4, 0.2).

This script re-trains the headline agent (Double DQN) under a grid of weight
triplets that sum to 1.0 and reports, for each triplet, the evaluation reward,
mean comfort, mean CBT and — crucially — the fraction of *no-intervention*
(action 0) decisions. It quantifies how strongly the "withhold cooling" policy
depends on the energy penalty weight.

All runs use a fixed seed so the table is reproducible.

Usage:
    python reproducibility/sensitivity_reward_weights.py --seed 42
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from streamlit_dqn import (  # noqa: E402
    MmCowsDataLoader, HeatStressEnv, DQNAgent, DQNConfig, train_episode,
)

# (milk alpha, comfort beta, energy gamma) — each sums to 1.0
WEIGHT_GRID = [
    (0.40, 0.40, 0.20),   # <- published configuration
    (0.45, 0.45, 0.10),   # low energy penalty
    (0.50, 0.40, 0.10),
    (0.34, 0.33, 0.33),   # balanced
    (0.30, 0.30, 0.40),   # high energy penalty
    (0.25, 0.25, 0.50),   # very high energy penalty
    (0.30, 0.50, 0.20),   # comfort-weighted
    (0.20, 0.60, 0.20),   # strongly comfort-weighted
]


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def train_eval(data, weights, n_train, n_eval):
    rw = {"milk": weights[0], "comfort": weights[1], "energy": weights[2]}
    env = HeatStressEnv(data=data, episode_length=48, reward_weights=rw)
    cfg = DQNConfig(state_dim=7, action_dim=4, use_double_dqn=True,
                    use_dueling=False, use_per=False, use_noisy=False)
    agent = DQNAgent(cfg)
    for ep in range(n_train):
        train_episode(agent, env, ep)
    rewards, comfort, cbt, actions = [], [], [], []
    for _ in range(n_eval):
        s, _ = env.reset(); ep_r, ep_c, ep_cbt = 0.0, [], []
        while True:
            a = agent.select_action(s, training=False)
            s, r, term, trunc, info = env.step(a)
            ep_r += r; ep_c.append(info.get("comfort", 0.5)); ep_cbt.append(info.get("cbt", 38.5))
            actions.append(a)
            if term or trunc:
                break
        rewards.append(ep_r); comfort.append(np.mean(ep_c)); cbt.append(np.mean(ep_cbt))
    dist = np.bincount(actions, minlength=4) / len(actions)
    return dict(mean_reward=float(np.mean(rewards)), std_reward=float(np.std(rewards)),
                mean_comfort=float(np.mean(comfort)), mean_cbt=float(np.mean(cbt)),
                action0_pct=float(dist[0] * 100), action_distribution=[float(x) for x in dist])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--data", default="data/mmcows_real/sensor_data")
    ap.add_argument("--outdir", default="results/reproducibility")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    data = MmCowsDataLoader(str(root / args.data)).load()
    if data is None:
        print("ERROR: data not found"); return

    rows = []
    for w in WEIGHT_GRID:
        set_seed(args.seed)  # same seed for every grid point -> isolates weight effect
        res = train_eval(data, w, args.episodes, args.eval_episodes)
        tag = "published" if w == (0.40, 0.40, 0.20) else ""
        rows.append({"alpha_milk": w[0], "beta_comfort": w[1], "gamma_energy": w[2],
                     "mean_reward": round(res["mean_reward"], 3),
                     "std_reward": round(res["std_reward"], 3),
                     "mean_comfort": round(res["mean_comfort"], 3),
                     "mean_cbt": round(res["mean_cbt"], 3),
                     "no_intervention_pct": round(res["action0_pct"], 1),
                     "note": tag})
        print(f"  w={w} -> reward={res['mean_reward']:.2f} action0={res['action0_pct']:.1f}% cbt={res['mean_cbt']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "sensitivity_reward_weights.csv", index=False)
    with open(outdir / "sensitivity_reward_weights.json", "w") as f:
        json.dump({"seed": args.seed, "episodes": args.episodes,
                   "eval_episodes": args.eval_episodes, "grid": rows}, f, indent=2)
    print(f"\nWrote sensitivity table to {outdir}/sensitivity_reward_weights.csv")


if __name__ == "__main__":
    main()
