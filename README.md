# Satellite Image Classification via Federated Learning

Cross-region **federated learning** for satellite land-use/land-cover (LULC) classification —
a reproducible proof-of-concept in which many *regional* clients train a shared classifier
without pooling raw imagery.

> **This repo is mid-rebuild.** The first version produced invalid results; it was audited and is
> being rebuilt top-down for reproducibility and conference-grade honesty.
>
> - 🔍 [`previous-progress.md`](previous-progress.md) — audit of the old pipeline (19 defects, B1–B19)
> - 🗺️ [`PLAN.md`](PLAN.md) — the rebuild design (dataset, partitioning, method, evaluation, bug→fix map)
> - ✅ [`TODO.md`](TODO.md) — phase-by-phase execution checklist (P0–P7) with sanity gates

## Approach in one paragraph

Use **real EuroSAT** (27,000 Sentinel-2 patches, 10 classes) and **emulate** cross-region federation
by partitioning it across `K` clients with **Dirichlet(α)** label skew — no continent-specific data
is collected. Federated training runs in **Flower**; the proposed method is a **FedBN-based
personalized federated transfer-learning** scheme compared against centralized, local-only, FedAvg,
and FedProx baselines, under a *fixed, shared* evaluation protocol (same splits, test-only reporting,
global-model evaluation each round, multiple seeds with confidence intervals).

## How to run (Google Colab — no local GPU needed)

Everything runs on Colab; logic lives in the `fedsat` package and notebooks are thin drivers.

1. **Push this repo to GitHub** (the notebooks `git clone` it to fetch the code).
2. Open a notebook from `notebooks/` in Colab (via *File → Open notebook → GitHub*, or upload it).
3. Set **Runtime → Change runtime type → T4 GPU**.
4. **Run all.** The first cell clones the repo, installs `requirements.txt`, and imports `fedsat`.
   Cell 2 optionally mounts Google Drive so the data cache, partitions, and results persist.

Run the notebooks in order:

| Notebook | Phase | What it does |
|---|---|---|
| `notebooks/00_setup_and_eda.ipynb` | P0 | data integrity gate (G1), EDA, Dirichlet partition + disjointness gate (G2), save partition |
| `notebooks/01_centralized_baseline.ipynb` | P1 | centralized ResNet-18 upper bound; overfit-one-batch gate (G3); full test metrics |
| `notebooks/02_federated_fedavg.ipynb` | P2 | FedAvg (transparent core, evaluates global model each round) + **G4: FedAvg ≈ centralized on IID**; optional real Flower parity |
| `notebooks/03_noniid_sweep.ipynb` | P3 | resumable Dirichlet-α sweep comparing centralized / local-only / FedAvg / FedProx |
| `notebooks/04_proposed_pftl.ipynb` | P4 | proposed FedBN vs FedAvg/FedProx/GroupNorm under per-client sensor shift; BN-policy + shift ablations, multi-seed |
| `notebooks/05_scale_and_comm.ipynb` | P5 | scale (K∈{5,10,20,50}) + partial participation + uplink compression (top-k / 8-bit), accuracy-per-MB |
| `notebooks/06_loco_generalization.ipynb` | P6 | leave-one-region-out generalization to an unseen sensor + label-free AdaBN adaptation |
| `notebooks/07_analysis_figures.ipynb` | P7 | **CPU-only** analysis: master table, paired stats, all figures, report §5–§8 (reads saved results, no GPU/dataset) |

Notebooks 00 and 01 share the same `dataset / num_clients / alpha / seed` so 01 loads the exact
partition saved by 00.

## Repository map

```
PLAN.md, TODO.md, previous-progress.md   # design, checklist, audit
requirements.txt                          # Colab-friendly deps (torch ships with Colab)
src/fedsat/
  config.py    # typed ExperimentConfig (YAML round-trip)
  utils.py     # seeding, hashing, device, provenance
  data.py      # EuroSAT loader + integrity gate + Dirichlet partition + splits + transforms + SensorShift
  models.py    # ResNet-18/50 builder, norm policy, multispectral stem
  engine.py    # centralized/local train + full-metric evaluation
  fl.py        # FedAvg / FedProx / FedBN + uplink compression (run_fedavg / run_federated)
  regimes.py   # centralized + local-only baseline runners
notebooks/     # thin Colab drivers (00, 01, 02, 03, 04, …)
```

## License

See [LICENSE](LICENSE).
