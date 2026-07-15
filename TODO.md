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

- [x] Add **FedProx** (proximal μ) to the FL core — `fedsat/fl.py` (`run_fedavg(..., mu=...)`, tested)
- [x] Add **centralized** + **local-only** regime runners — `fedsat/regimes.py` (tested)
- [x] Resumable sweep over α ∈ {100, 1.0, 0.5, 0.1}, K=10 (partitions cached; combos skipped if done)
- [x] Regimes {centralized, local-only, FedAvg, FedProx}; local-only reports global-test + own-test
- [x] Plot accuracy vs α; worst-client; save sweep tables (CSV)
- [ ] **RUN on Colab** (resumable; ~few hours, 1 seed) ← *next action for you*
- [ ] Scale `SEEDS` to 3 for confidence intervals (final paper)
- [ ] **Exit:** "accuracy degrades as α↓" reproduced; local-only global-test < FedAvg under skew

## P4 — Proposed method: Personalized FTL  ·  notebook `04_proposed_pftl.ipynb`  (E4, E5)

- [x] Implement **FedBN** (keep BN local, exclude from aggregation) + unified per-client `run_federated` — `fedsat/fl.py` (kills B6, tested)
- [x] FedProx term composes with FedBN (FedBN+Prox); GroupNorm BN-free control via `norm='gn'`
- [x] Per-client **sensor-shift** simulation (fixed photometric/atmospheric per client) — `fedsat/data.py` `SensorShift`/`build_client_shifts` (Track A+, tested)
- [x] Metric = **mean per-client own-test accuracy** (defined for FedBN); val-based selection
- [x] E4 BN-policy ablation {aggregate-BN, FedBN, GroupNorm} + E5 shift on/off, resumable, multi-seed
- [x] Honest verdict cell (claim a win only where mean±std separates)
- [x] **RAN on Colab — 3 seeds complete.** Under sensor shift: **FedBN 0.910±0.016 vs FedAvg 0.839±0.007 (+7.1pt) / FedProx 0.852±0.011 (+5.8pt)**, non-overlapping; per-seed gaps all positive (+5.3/+6.6/+9.4pt). GroupNorm control 0.784 confirms *personalized* BN is the mechanism; FedBN+Prox≈FedBN (proximal adds nothing). Honest cost: ~1.6pt when no shift.
- [x] **Exit MET:** FedBN's advantage under feature shift isolated with separated CIs — this is the contribution.

## P5 — Scale & communication  ·  notebook `05_scale_and_comm.ipynb`  (E3, E8)

- [x] E3 scale: K ∈ {5,10,20,50} (full participation) — accuracy + total comm vs K
- [x] E3 partial participation: `fraction_fit` ∈ {0.2,0.5,1.0} at K=20
- [x] E8 real uplink compression: none / top-k (10%,1%) / 8-bit in `fedsat/fl.py` (`run_fedavg(compress=...)`, tested) — accuracy-per-MB curve (kills B14 claim gap)
- [x] Resumable per `(K, fraction_fit, compression, seed)`; runs deduplicated across studies
- [x] **RAN on Colab (num_workers=0 RAM fix).** Scale K5→50: 0.968→0.937, comm linear (6.4→64GB). Participation K=20: ff1.0 0.955 / ff0.5 0.943 (½ comm) / ff0.2 0.845. Compression K=10: 8-bit 0.967 (4× less uplink, lossless), top-k10% 0.949 (5×), top-k1% 0.871 (50×).
- [x] Compression figure fixed to report **uplink** (downlink is full & dominates total) — recomputed post-hoc, no re-run
- [x] **Exit MET:** scale + partial-participation + honest communication trade-off curves produced

## P6 — Cross-domain generalization  ·  notebook `06_loco_generalization.ipynb`  (E7, opt. E6, E9)

- [x] E7 leave-one-client-out: `run_loco` in `fedsat/fl.py` — train global on K−1 regions, eval on unseen region test (tested)
- [x] **AdaBN** test-time BN re-estimation on the unseen region's unlabeled data (`_recompute_bn`) — label-free feature-shift fix, ties to P4 BN theme
- [x] K=5 leave-one-region-out × {FedAvg, FedProx} × {base, +AdaBN}, under sensor shift; resumable; per-region + averaged results
- [x] **RAN on Colab (10 runs, 5 regions × 2 methods).** FedAvg base 0.748 on unseen region (4/5 regions 0.73–0.88 out-of-the-box). **AdaBN: mean +5pt, worst region 0.45→0.71 (+26pt), variance halved (0.18→0.08)** — rescues the hard region; mildly hurts already-easy regions (2,3). BN stats are the feature-shift lever (FedBN participating, AdaBN unseen).
- [ ] E6 (optional) multispectral EuroSAT_MSI; E9 (optional, Track B) multi-dataset — deferred unless requested
- [x] **Exit MET:** unseen-region generalization quantified; AdaBN worst-case/stability recovery shown

## P7 — Analysis, figures, writeup  ·  notebook `07_analysis_figures.ipynb`  (CPU-only, no GPU/dataset)

- [x] Reads P1–P6 saved result files from Drive (defensive: skips phases not yet run)
- [x] Master results table (per-phase headline; P4 with mean ± 95% CI) → `master_results.csv`
- [x] Paired significance test for the P4 FedBN claim (per-seed diffs, 95% CI, paired t-test)
- [x] Figures from saved data: α-curve, FedBN ablation, scale/participation/compression, LOCO+AdaBN, P1 confusion/per-class
- [x] Auto-emits report-ready §5–§8 markdown with real numbers + honest limitations → `report_results.md`
- [ ] **RUN on Colab CPU** after P6 finishes (re-run anytime; reflects current Drive results) ← *next*
- [ ] **Exit:** every figure/number traces to a run under `results/`

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
| `02_federated_fedavg.ipynb` (P2) | ✅ **G4 PASSED on Colab** | FedAvg IID 0.9765 vs centralized 0.9573; Flower integration verified (Ray backend blocked by Colab TF/protobuf pin — noted) |
| `03_noniid_sweep.ipynb` (P3) | ✅ **RAN on Colab** | FedAvg 0.975→0.845 as α↓; FedProx +7pt at α=0.1; local-only global collapses 0.94→0.40 |
| `04_proposed_pftl.ipynb` (P4) | ✅ **DONE (3 seeds, CIs)** | under shift: FedBN 0.910±0.016 vs FedAvg 0.839±0.007 / FedProx 0.852±0.011 — non-overlapping (+7/+6pt). GroupNorm control 0.784. Contribution validated |
| `05_scale_and_comm.ipynb` (P5) | ✅ **RAN on Colab** | scale acc 0.968→0.937 (K 5→50), comm linear; ff=0.5 halves comm for ~1pt; 8-bit lossless (4× less uplink), top-k 1% 50× uplink for −9.5pt. RAM fixed via num_workers=0 |
| `06_loco_generalization.ipynb` (P6) | ✅ **RAN on Colab** | unseen-region FedAvg base 0.748; AdaBN worst region 0.45→0.71, variance halved (rescues hard region, mild cost on easy ones) |
| `07_analysis_figures.ipynb` (P7) | ✅ **ready to run (CPU-only)** | master table + paired stats + all figures + report §5–§8 md; reads saved results, no GPU/dataset |

> **P2 design note:** FedAvg runs in two layers — a **transparent, tested `run_fedavg` core** that
> clears G4 reliably (no dependence on Flower version), plus an **optional pinned Flower parity**
> section (defensive) so the write-up can cite Flower. Run α=100 (IID) first; G4 requires FedAvg
> within ~3% of the 0.957 centralized baseline.

> **P1 baseline recorded 2026-07-11:** centralized ResNet-18 on EuroSAT = **95.7% test acc** (macro-F1
> 0.957). This is the upper bound the FL regimes are measured against. Per-class F1 balanced 0.90–0.99
> (contrast old project: Industrial 0.05 / River 0.03 — collapse fixed).

> **Action required to run in Colab:** commit & push these new files to GitHub `main` — the
> notebooks `git clone` the repo to fetch the `fedsat` package.
