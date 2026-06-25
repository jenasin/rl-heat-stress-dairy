# Reproducibility package

Companion code and logs for the manuscript *"Deep reinforcement learning for
simulated autonomous heat stress mitigation in dairy cattle"*, prepared in
response to the reviewer's reproducibility requests.

**Start here:** [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — answers every
reviewer placeholder (seeds, split, repeated runs, hardware, missingness,
sensitivity) and documents exactly what was run. No reported table number is
changed.

## Contents

| File | Purpose |
|---|---|
| `REPRODUCIBILITY.md` | Full reproducibility report (read this first) |
| `latex_snippets.tex` | Ready-to-paste LaTeX filling the reviewer's `[insert…]` gaps |
| `reproduce.py` | Seeded multi-seed re-run; saves results, models, `run_manifest.json` |
| `sensitivity_reward_weights.py` | Reward-weight sensitivity analysis |
| `make_figures.py` | Re-renders Figures 1, 3, 4 (published numbers) + suppl. |
| `make_fig2_training.py` | Re-renders Figure 2 (training curves) |
| `requirements.txt` | Pinned reproduction environment |

## Generated outputs (in `../results/reproducibility/`)

- `results_seed*.json` — per-seed evaluation results (5 seeds)
- `results_multiseed.json` — mean ± SD across seeds
- `run_manifest.json` — seeds, hardware, versions, split, missingness
- `sensitivity_reward_weights.csv` — sensitivity table
- `models/*.pt` — saved network checkpoints
- `figures/` — re-rendered manuscript figures (PNG + PDF)

## The exact code behind the published numbers

Lives one directory up: `streamlit_dqn.py` (environment, agents, reward) and
`run_experiment.py` (driver). `../results/paper_figures/results.json` matches
Table 3 to four decimals. This package wraps that code for reproducibility; it
does not modify it.
