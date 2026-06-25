#!/usr/bin/env python3
"""
Re-render the manuscript figures addressing the reviewer's notes.
=================================================================

Reviewer notes addressed (WITHOUT changing any reported number):
  * Figure 1: explicit axis labels/ticks on every panel, panel letters (a)-(d),
    and standard-deviation bands on the diurnal panel (the caption already
    promises "with standard deviation bands"), no cropping.
  * Figure 3: non-overlapping method labels; SD error bars. A faithful version
    using the PUBLISHED means (reward SD from results.json) is written as
    ``comparison_main`` and a multi-seed version with SD on all three panels as
    ``comparison_multiseed``.
  * Figure 4: panel left intact (published action distribution), no cropping.

Figure 1 uses the same preprocessed dataset; Figures 3/4 use the published
``results/paper_figures/results.json`` so the bar heights are byte-identical to
the article tables.

Usage:
    python reproducibility/make_figures.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Match the original article's polished look (seaborn whitegrid + tab10 palette).
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "axes.titleweight": "bold"})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from streamlit_dqn import MmCowsDataLoader  # noqa: E402

# Per-method colours matching the original figures (tab10).
METHOD_COLOR = {
    "No Action": "#9467bd", "THI Threshold": "#8c564b", "CBT Threshold": "#e377c2",
    "Random": "#7f7f7f", "Standard DQN": "#1f77b4", "Double DQN": "#ff7f0e",
    "Dueling DQN": "#2ca02c", "Double Dueling DQN + PER": "#d62728",
}

ROOT = Path(__file__).resolve().parent.parent
PUB = ROOT / "results" / "paper_figures" / "results.json"
MULTISEED = ROOT / "results" / "reproducibility" / "results_multiseed.json"
OUT = ROOT / "results" / "reproducibility" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = ["No Action", "THI Threshold", "CBT Threshold", "Random",
                "Standard DQN", "Double DQN", "Dueling DQN", "Double Dueling DQN + PER"]
SHORT = {"No Action": "No Action", "THI Threshold": "THI Thr.", "CBT Threshold": "CBT Thr.",
         "Random": "Random", "Standard DQN": "Std DQN", "Double DQN": "Double DQN",
         "Dueling DQN": "Dueling DQN", "Double Dueling DQN + PER": "DD-DQN+PER"}


def _key(results, name):
    """Published results.json uses slightly different casing for thresholds."""
    for k in results:
        if k.lower().replace(" ", "") == name.lower().replace(" ", ""):
            return k
    aliases = {"THI Threshold": "Thi Threshold", "CBT Threshold": "Cbt Threshold"}
    return aliases.get(name, name)


def figure1_data_overview():
    data = MmCowsDataLoader(str(ROOT / "data/mmcows_real/sensor_data")).load()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.hist(data["thi"], bins=30, color="#ff7f0e", edgecolor="black", alpha=0.75)
    ax.axvline(72, color="red", ls="--", lw=2, label="Heat-stress threshold (THI = 72)")
    ax.set_xlabel("Temperature-Humidity Index (THI)"); ax.set_ylabel("Frequency")
    ax.set_title("(a) THI distribution", loc="left", fontweight="bold"); ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.hist(data["cbt"], bins=30, color="#d62728", edgecolor="black", alpha=0.75)
    ax.axvline(39.0, color="darkred", ls="--", lw=2, label="Fever threshold (39 °C)")
    ax.set_xlabel("Core body temperature (°C)"); ax.set_ylabel("Frequency")
    ax.set_title("(b) CBT distribution", loc="left", fontweight="bold"); ax.legend(fontsize=9)

    ax = axes[1, 0]
    sc = ax.scatter(data["thi"], data["cbt"], c=data["hour"], cmap="twilight", alpha=0.5, s=10)
    ax.set_xlabel("THI"); ax.set_ylabel("CBT (°C)")
    ax.set_xlim(data["thi"].min() - 1, data["thi"].max() + 1)
    ax.set_ylim(data["cbt"].min() - 0.1, data["cbt"].max() + 0.1)
    ax.set_title("(c) THI vs CBT (coloured by hour)", loc="left", fontweight="bold")
    cb = plt.colorbar(sc, ax=ax, label="Hour of day"); cb.set_ticks(range(0, 25, 6))

    ax = axes[1, 1]
    g = data.groupby("hour")
    h = sorted(data["hour"].unique())
    thi_m, thi_s = g["thi"].mean(), g["thi"].std()
    cbt_m, cbt_s = g["cbt"].mean(), g["cbt"].std()
    ax2 = ax.twinx()
    l1, = ax.plot(thi_m.index, thi_m.values, "o-", color="#ff7f0e", lw=2, label="THI")
    ax.fill_between(thi_m.index, (thi_m - thi_s).values, (thi_m + thi_s).values,
                    color="#ff7f0e", alpha=0.2)
    l2, = ax2.plot(cbt_m.index, cbt_m.values, "s-", color="#d62728", lw=2, label="CBT")
    ax2.fill_between(cbt_m.index, (cbt_m - cbt_s).values, (cbt_m + cbt_s).values,
                     color="#d62728", alpha=0.2)
    ax.set_xlabel("Hour of day"); ax.set_xticks(range(0, 25, 3))
    ax.set_ylabel("THI", color="#ff7f0e"); ax2.set_ylabel("CBT (°C)", color="#d62728")
    ax.set_title("(d) Diurnal patterns (mean ± SD)", loc="left", fontweight="bold")
    ax.legend([l1, l2], ["THI", "CBT"], loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "data_overview.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "data_overview.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Figure 1 (data_overview) written")


def figure3_comparison():
    results = json.load(open(PUB))
    methods = [m for m in METHOD_ORDER if _key(results, m) in results]
    rew = [results[_key(results, m)]["mean_reward"] for m in methods]
    rew_sd = [results[_key(results, m)]["std_reward"] for m in methods]
    com = [results[_key(results, m)]["mean_comfort"] for m in methods]
    cbt = [results[_key(results, m)]["mean_cbt"] for m in methods]
    x = np.arange(len(methods))
    colors = [METHOD_COLOR.get(m, "#4ecdc4") for m in methods]
    labels = [SHORT[m] for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].bar(x, rew, yerr=rew_sd, capsize=4, color=colors, edgecolor="black")
    axes[0].set_ylabel("Mean reward"); axes[0].set_title("(a) Reward (mean ± SD)", loc="left", fontweight="bold")
    axes[1].bar(x, com, color=colors, edgecolor="black")
    axes[1].set_ylabel("Comfort score"); axes[1].set_ylim(0.75, 0.9)
    axes[1].set_title("(b) Comfort", loc="left", fontweight="bold")
    axes[2].bar(x, cbt, color=colors, edgecolor="black")
    axes[2].axhline(39.0, color="red", ls="--", label="Fever threshold (39 °C)")
    axes[2].set_ylabel("Mean CBT (°C)"); axes[2].set_ylim(38.0, 39.1)
    axes[2].set_title("(c) Mean CBT", loc="left", fontweight="bold"); axes[2].legend(fontsize=9)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "comparison_main.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "comparison_main.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Figure 3 (comparison_main, published numbers) written")


def figure3_multiseed():
    if not MULTISEED.exists():
        print("multiseed results not ready; skipping comparison_multiseed")
        return
    agg = json.load(open(MULTISEED))
    methods = [m for m in METHOD_ORDER if m in agg]
    x = np.arange(len(methods))
    colors = [METHOD_COLOR.get(m, "#4ecdc4") for m in methods]
    labels = [SHORT[m] for m in methods]
    rew = [agg[m]["reward_mean"] for m in methods]; rew_sd = [agg[m]["reward_sd"] for m in methods]
    com = [agg[m]["comfort_mean"] for m in methods]; com_sd = [agg[m]["comfort_sd"] for m in methods]
    cbt = [agg[m]["cbt_mean"] for m in methods]; cbt_sd = [agg[m]["cbt_sd"] for m in methods]
    n = agg[methods[0]]["n_seeds"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].bar(x, rew, yerr=rew_sd, capsize=4, color=colors, edgecolor="black")
    axes[0].set_ylabel("Mean reward"); axes[0].set_title("(a) Reward", loc="left", fontweight="bold")
    axes[1].bar(x, com, yerr=com_sd, capsize=4, color=colors, edgecolor="black")
    axes[1].set_ylabel("Comfort score"); axes[1].set_ylim(0.75, 0.9)
    axes[1].set_title("(b) Comfort", loc="left", fontweight="bold")
    axes[2].bar(x, cbt, yerr=cbt_sd, capsize=4, color=colors, edgecolor="black")
    axes[2].axhline(39.0, color="red", ls="--", label="Fever threshold (39 °C)")
    axes[2].set_ylabel("Mean CBT (°C)"); axes[2].set_ylim(38.0, 39.1)
    axes[2].set_title("(c) Mean CBT", loc="left", fontweight="bold"); axes[2].legend(fontsize=9)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(f"Supplementary: mean ± SD across {n} independent seeds", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "comparison_multiseed.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "comparison_multiseed.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Supplementary comparison_multiseed (n={n} seeds) written")


def figure4_action_distribution():
    results = json.load(open(PUB))
    action_names = ["No\nintervention", "Fans", "Fans +\nsprinklers", "Water\nspray"]
    rl = [m for m in METHOD_ORDER if _key(results, m) in results
          and "action_distribution" in results[_key(results, m)]]
    x = np.arange(len(action_names)); width = 0.8 / len(rl)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, m in enumerate(rl):
        dist = np.array(results[_key(results, m)]["action_distribution"])
        off = (i - len(rl) / 2 + 0.5) * width
        ax.bar(x + off, dist * 100, width, label=SHORT[m], color=colors[i % 4])
    ax.set_xticks(x); ax.set_xticklabels(action_names, fontsize=10)
    ax.set_ylabel("Action frequency (%)")
    ax.set_title("Action distribution by RL agent (evaluation)", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "action_distribution.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "action_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Figure 4 (action_distribution) written")


if __name__ == "__main__":
    figure1_data_overview()
    figure3_comparison()
    figure3_multiseed()
    figure4_action_distribution()
    print(f"\nAll figures written to {OUT}")
