# Deep reinforcement learning for simulated autonomous heat stress mitigation in dairy cattle

Code and reproducibility material accompanying the manuscript:

> Saro J., Ducháček J., Mazancová J., Stádník L., Malinovský V.
> *Deep reinforcement learning for simulated autonomous heat stress mitigation
> in dairy cattle.* (under review).

This is a **simulation-based proof of concept**: cooling interventions were not
applied to animals; their expected effects were computationally modelled as
action-specific reductions in effective thermal load with energy penalties.
Four DQN variants (Standard, Double, Dueling, Double Dueling + Prioritized
Experience Replay) are trained and evaluated on the real **MmCows** multi-modal
dairy-cattle sensor dataset.

## Repository layout

```
├── streamlit_dqn.py          # environment, DQN agents, networks, reward, data loader
├── run_experiment.py         # driver that produced the published Table 3 + Figures 1–4
├── run_experiments.py        # auxiliary Stable-Baselines3 (PPO/DQN) experiments
├── main.py, train*.py        # additional entry points
├── src/                      # supporting modules
├── configs/                  # experiment configuration
├── results/
│   ├── paper_figures/        # PUBLISHED figures + results.json (matches Table 3 to 4 dp)
│   └── reproducibility/      # seeded multi-seed outputs, models, manifest, re-rendered figures
└── reproducibility/          # reproducibility scripts + documentation  ← START HERE
```

## Reproducibility

See [`reproducibility/REPRODUCIBILITY.md`](reproducibility/REPRODUCIBILITY.md)
for the full account (random seeds, train/eval protocol, repeated runs,
hardware, modality missingness, reward-weight sensitivity). The
corresponding manuscript text changes are kept with the manuscript itself
(not in this code repository).

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python reproducibility/reproduce.py --seeds 42 123 456 789 1011   # multi-seed runs
python reproducibility/sensitivity_reward_weights.py --seed 42    # reward-weight sensitivity
python reproducibility/make_figures.py                            # re-render Figures 1, 3, 4
python reproducibility/make_fig2_training.py                      # re-render Figure 2
```

### Headline result

The Double DQN agent achieved the highest numerical reward (18.26 ± 3.89) but
its learned policy selected **no intervention** in essentially all evaluation
cases and was not statistically different from the No-Action baseline
(17.66 ± 5.10; Welch t-test, p = 0.510). The seeded 5-seed reproduction and the
reward-weight sensitivity analysis confirm this finding (see
`reproducibility/REPRODUCIBILITY.md`).

## Data availability

The MmCows dataset is publicly available and is **not redistributed here**.
Download it and place it under `data/`:

- HuggingFace: <https://huggingface.co/datasets/neis-lab/mmcows>
- Project: <https://github.com/neis-lab/mmcows>

> Reference: Xu T., Zhang Y., Zhu M., et al. *MmCows: a multimodal dataset for
> dairy cattle monitoring.* NeurIPS 2024.

## Notes

- The raw dataset (`data/`) and the Python virtual environment (`venv/`) are
  excluded from version control.
- The original published run did not set a global random seed; the seeded
  scripts under `reproducibility/` provide reproducible repeated runs that
  confirm the qualitative findings without altering the published numbers.

## Contact

Jan Saro — Czech University of Life Sciences Prague — saroj@pef.czu.cz
