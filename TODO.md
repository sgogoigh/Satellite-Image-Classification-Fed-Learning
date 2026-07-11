# TODO — Rebuild execution checklist

> Working checklist for executing [PLAN.md](PLAN.md). Organized by the plan's phases (P0–P7).
> Each phase lists its **exit gate** (from PLAN §12) — do not start the next phase until it is green.
> Legend: `[ ]` todo · `[~]` in progress · `[x]` done · 🟢 = automated sanity gate.
>
> **Compute:** all runs on **Google Colab GPU** (no local GPU). Code lives in `src/fedsat/` and is
> imported by thin notebooks in `notebooks/`. See [README.md](README.md) for the Colab workflow.

---

## Legend of deliverables per phase

| Phase | Notebook | Depends on | Exit gate |
|---|---|---|---|
| P0 Setup & data | `00_setup_and_eda.ipynb` | — | 🟢 G1, G2 |
| P1 Centralized baseline | `01_centralized_baseline.ipynb` | P0 partitions | 🟢 G3 + ~95% |
| P2 FL core (Flower) | `02_federated_fedavg.ipynb` | P1 | 🟢 **G4**, G5 |
| P3 Non-IID sweep | `03_noniid_sweep.ipynb` | P2 | α-curve reproduced |
| P4 Proposed PFTL | `04_proposed_pftl.ipynb` | P3 | ablations + CIs |
| P5 Scale & comm | `05_scale_and_comm.ipynb` | P2 | scale + comm curves |
| P6 Generalization | `06_loco_generalization.ipynb` | P4 | LOCO quantified |
| P7 Analysis & writeup | `07_analysis_figures.ipynb` | P2–P6 | every claim traced |

---

## P0 — Environment & data  ·  notebook `00_setup_and_eda.ipynb`

- [ ] Colab setup cell: clone repo (or mount Drive), `pip install -r requirements.txt`, add `src/` to path
- [ ] Verify env: `import torch`, CUDA available, print versions (kills B17)
- [ ] Implement `ExperimentConfig` (YAML round-trip) — `src/fedsat/config.py`
- [ ] Implement seeding + hashing utils — `src/fedsat/utils.py`
- [ ] Implement EuroSAT loader reading labels from the dataset's own `ClassLabel` (kills B3) — `src/fedsat/data.py`
- [ ] 🟢 **G1 Integrity gate**: assert real data, exactly 10 classes, every class count > 0, total ≈ 27,000; record data hash; **no synthetic code path exists** (kills B1, B2)
- [ ] EDA: class-distribution bar chart, sample-image grid, per-channel RGB stats
- [ ] Implement Dirichlet partitioner (numpy, deterministic, saves indices) — `src/fedsat/data.py`
- [ ] Build split: global stratified `test` (comparable across regimes) + `pool`; Dirichlet-partition `pool` across K clients; per-client stratified train/val/test
- [ ] Persist partition indices to Drive: `data/partitions/{dataset}_{K}_alpha{α}_seed{seed}.json` (reused by every regime — kills B7, B8)
- [ ] 🟢 **G2 Disjoint splits**: assert global-test ∩ clients = ∅; per-client train/val/test disjoint; clients non-overlapping
- [ ] Partition diagnostics: class×client heatmap, per-client counts, per-client label entropy
- [ ] **Exit:** G1 + G2 green; partition file saved and reloadable

## P1 — Centralized baseline  ·  notebook `01_centralized_baseline.ipynb`

- [ ] Implement model builder (ResNet-18/50, ImageNet pretrained, norm policy, MS stem hook) — `src/fedsat/models.py`
- [ ] Implement train/eval engine: `train_one_epoch`, `evaluate` (acc, macro-F1, per-class F1, confusion, κ — all from the same model on TEST, kills B10), `fit` w/ early stopping — `src/fedsat/engine.py`
- [ ] 🟢 **G3 Overfit-one-batch**: model reaches ~100% on a single batch (learning code correct)
- [ ] Train centralized on **union of client train** splits; early-stop on **union of client val**; SGD+momentum, LR schedule (kills B5 optimizer issue at baseline)
- [ ] Evaluate on the **global test set** and report full metric suite (test-only, B7/B10)
- [ ] Save results (`results/<run_id>/`) + checkpoint + config + data hash + git commit (kills B19)
- [ ] **Exit:** centralized test accuracy in the ~95%+ EuroSAT ballpark; G3 green

## P2 — Federated core  ·  notebook `02_federated_fedavg.ipynb`  ← make-or-break

- [x] Implement transparent FedAvg core (SGD clients, few local epochs) — `src/fedsat/fl.py` (tested)
- [x] Run FedAvg over the **IID** partition (α=100)
- [x] 🟢 **G5 Global-eval**: round metrics come from the aggregated **global** model on the global test set (kills B4) — done in `run_fedavg`
- [x] Log real communication MB/round; persist round history/curves (kills B14, B18)
- [x] 🟢 **G4 FedAvg ≡ Centralized on IID** check wired (within ~3% of P1's 0.957)
- [x] Optional **real Flower parity** section (pinned `flwr[simulation]`, `ClientApp`/`ServerApp`/`run_simulation`, defensive) — cite Flower (kills B12)
- [ ] **RUN on Colab** and confirm G4 passes ← *next action for you*
- [ ] If Flower parity errors, send `flwr.__version__` so the cell can be finalized to that API
- [ ] **Exit:** **G4 green** on Colab (if not, stop and debug before P3)

## P3 — Non-IID heterogeneity sweep  ·  notebook `03_noniid_sweep.ipynb`  (E2)

- [ ] Generate partitions for α ∈ {100, 1.0, 0.5, 0.1}, K=10 (P0 partitioner)
- [ ] Run regimes {centralized, local-only, FedAvg, FedProx} across α, ≥3 seeds
- [ ] Add FedProx strategy (proximal μ)
- [ ] Plot accuracy vs α; per-client (worst-client) accuracy; convergence curves
- [ ] **Exit:** expected "accuracy degrades as α↓" curve reproduced; local-only < FedAvg on skew

## P4 — Proposed method: Personalized FTL  ·  notebook `04_proposed_pftl.ipynb`  (E4, E5)

- [ ] Implement **FedBN** (keep BN layers local; exclude from aggregation) — custom Flower strategy (kills B6)
- [ ] Implement optional personalized classifier head; optional FedProx term
- [ ] Implement per-client **sensor-shift** simulation (seeded photometric/atmospheric transforms) — `src/fedsat/data.py` (Track A+)
- [ ] E4 BN-policy ablation: {aggregate-BN, FedBN, GroupNorm} × {FedAvg, PFTL}, ≥5 seeds
- [ ] E5 feature-shift ablation: shift on/off × {FedAvg, FedProx, PFTL}, ≥5 seeds
- [ ] Compute mean ± 95% CI; paired Wilcoxon/t-test PFTL vs baselines — `src/fedsat/eval/stats.py`
- [ ] **Exit:** each PFTL component's contribution isolated; claims only where CIs separate (honest reporting per PLAN §6.2)

## P5 — Scale & communication  ·  notebook `05_scale_and_comm.ipynb`  (E3, E8)

- [ ] E3 scale: K ∈ {5,10,20,50} × `fraction_fit` ∈ {0.2,0.5,1.0} (demonstrates continent-scale feasibility)
- [ ] E8 communication: none / top-k sparsification / 8-bit quantization; accuracy-per-MB curves (kills B14 claim gap)
- [ ] **Exit:** scale demonstrated with partial participation; communication trade-off curves produced

## P6 — Cross-domain generalization  ·  notebook `06_loco_generalization.ipynb`  (E7, opt. E6, E9)

- [ ] E7 leave-one-client-out: train on K−1 clients, evaluate global model on unseen client test — `src/fedsat/eval/loco.py`
- [ ] E6 (optional) multispectral: EuroSAT_MSI 13-band + MS stem; RGB vs MS (kills B15 or delete claim)
- [ ] E9 (optional, Track B) multi-dataset cross-domain: EuroSAT+AID+NWPU+UC-Merced under shared taxonomy — `src/fedsat/data/taxonomy.py`; leave-one-domain-out
- [ ] **Exit:** participating vs unseen-client generalization gap quantified

## P7 — Analysis, figures, writeup  ·  notebook `07_analysis_figures.ipynb`

- [ ] Aggregate all runs over seeds; master results table with mean ± 95% CI
- [ ] Figures: accuracy-vs-α, convergence-vs-round, per-region bars, communication, confusion matrices
- [ ] 🟢 **G7 Budget parity** and 🟢 **G8 No-leak diagnostics** verified across the final result set
- [ ] Rewrite report §5–§8 from the **new** numbers; remove every unsupported claim; write honest limitations (PLAN §16)
- [ ] **Exit:** every figure/number in the writeup traces to a run under `results/`

---

## Cross-cutting engineering (do alongside, not a separate phase)

- [ ] `requirements.txt` pinned for Colab; `pyproject.toml`; `import torch` verified (B17)
- [ ] `src/fedsat/` package importable in Colab (clone or Drive); thin notebooks
- [ ] Seeding everywhere; deterministic partitions saved to disk; data + git hash per run (B19)
- [ ] Single canonical results store (`results/<run_id>/`), never stray CSVs (B19)
- [ ] `tests/`: partition-disjoint, integrity-gate, overfit-one-batch, fedavg≈centralized-iid, reproducibility
- [ ] Delete from old tree: `SyntheticRegionalDataset`, `DomainAwareAggregator`, CORAL self-align, probe per-class/κ, stray `results*.csv`, Flower stub (PLAN §15)

---

## Status snapshot (update as we go)

| Item | State | Notes |
|---|---|---|
| PLAN.md | ✅ done | design + bug traceability |
| previous-progress.md | ✅ done | audit of old pipeline |
| TODO.md | ✅ this file | |
| `src/fedsat/` package | ✅ built | config/utils/data/models/engine; partition + gate logic unit-tested locally |
| `00_setup_and_eda.ipynb` (P0) | ✅ **DONE (Colab)** | G1+G2 passed; partition `K10_alpha0.5_seed42` saved; artifacts in `outputs/P0-1/` |
| `01_centralized_baseline.ipynb` (P1) | ✅ **DONE (Colab)** | G3 passed; **centralized test acc 0.9573, macro-F1 0.9567, κ 0.9525**; no class collapse |
| `02_federated_fedavg.ipynb` (P2) | ✅ **ready to run** | transparent FedAvg core (`fedsat/fl.py`, tested) + **G4** gate + optional real Flower parity |
| P3–P7 notebooks | ⏳ pending | after G4 is green on Colab |

> **P2 design note:** FedAvg runs in two layers — a **transparent, tested `run_fedavg` core** that
> clears G4 reliably (no dependence on Flower version), plus an **optional pinned Flower parity**
> section (defensive) so the write-up can cite Flower. Run α=100 (IID) first; G4 requires FedAvg
> within ~3% of the 0.957 centralized baseline.

> **P1 baseline recorded 2026-07-11:** centralized ResNet-18 on EuroSAT = **95.7% test acc** (macro-F1
> 0.957). This is the upper bound the FL regimes are measured against. Per-class F1 balanced 0.90–0.99
> (contrast old project: Industrial 0.05 / River 0.03 — collapse fixed).

> **Action required to run in Colab:** commit & push these new files to GitHub `main` — the
> notebooks `git clone` the repo to fetch the `fedsat` package.
