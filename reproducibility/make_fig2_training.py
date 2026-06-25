#!/usr/bin/env python3
"""
Re-render Figure 2 (training curves) addressing the reviewer's note:
panels (c)/(d) without bottom cropping and panel (d) on the correct
observed-CBT scale. Curves are from a single seeded representative run
(seed 42); they are illustrative of convergence and contain no reported
table number.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"savefig.dpi": 300, "axes.titleweight": "bold"})

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from streamlit_dqn import (MmCowsDataLoader, HeatStressEnv, DQNAgent, DQNConfig, train_episode)  # noqa

OUT = ROOT / "results" / "reproducibility" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

CONFIGS = {
    "Standard DQN": dict(use_double_dqn=False, use_dueling=False, use_per=False, use_noisy=False),
    "Double DQN":   dict(use_double_dqn=True,  use_dueling=False, use_per=False, use_noisy=False),
    "Dueling DQN":  dict(use_double_dqn=True,  use_dueling=True,  use_per=False, use_noisy=False),
    "Double Dueling DQN + PER": dict(use_double_dqn=True, use_dueling=True, use_per=True, use_noisy=False),
}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def seed(s=42):
    import random
    random.seed(s); np.random.seed(s)
    import torch; torch.manual_seed(s)


def main(n_episodes=200):
    seed(42)
    data = MmCowsDataLoader(str(ROOT / "data/mmcows_real/sensor_data")).load()
    env = HeatStressEnv(data=data, episode_length=48)
    all_m = {}
    for name, flags in CONFIGS.items():
        cfg = DQNConfig(state_dim=7, action_dim=4, **flags)
        agent = DQNAgent(cfg)
        m = {k: [] for k in ("ep", "reward", "loss", "comfort", "cbt")}
        for ep in range(n_episodes):
            r = train_episode(agent, env, ep)
            m["ep"].append(ep); m["reward"].append(r["total_reward"])
            m["loss"].append(r["avg_loss"]); m["comfort"].append(r.get("avg_comfort", 0.5))
            m["cbt"].append(r.get("avg_cbt", 38.5))
        all_m[name] = m
        print(f"trained {name}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = [("reward", "Average reward", "(a) Training reward", False),
              ("loss", "Loss", "(b) Training loss", True),
              ("comfort", "Comfort score", "(c) Average comfort", False),
              ("cbt", "Core body temperature (°C)", "(d) Average CBT", False)]
    for ax, (key, ylab, title, logy) in zip(axes.flat, panels):
        for i, (name, m) in enumerate(all_m.items()):
            y = pd.Series(m[key]).rolling(20, min_periods=1).mean()
            ax.plot(m["ep"], y, label=name, color=COLORS[i], lw=2)
        ax.set_xlabel("Episode"); ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        if logy:
            ax.set_yscale("log")
        if key == "cbt":  # correct observed-CBT scale, no cropping
            ax.set_ylim(38.2, 38.8)
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "training_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 2 written to {OUT}")


if __name__ == "__main__":
    main()
