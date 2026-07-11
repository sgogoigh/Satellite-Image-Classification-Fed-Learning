# Previous Progress — Critical Review & Audit

> **Purpose of this document.** A top-to-bottom technical audit of the existing
> "Multi-Domain Federated Transfer Learning for Satellite Image Classification"
> capstone, written before we rebuild the project from scratch. It records *exactly*
> what was done, what the results really mean, where the implementation is broken or
> misleading, and what (if anything) is salvageable. Every claim below is traced to a
> specific notebook cell, output artifact, or executed log.
>
> **Reviewer stance:** research scientist preparing this work for a reproducible,
> conference-grade rebuild. The verdict is deliberately blunt.
>
> Date of audit: 2026-07-11.

---

## 0. Executive verdict

**The current results are not scientifically valid and must not be published as-is.**

The headline numbers reported in the thesis and `outputs/` —

| Regime | Accuracy | What it is measured on |
|---|---|---|
| Centralized | **0.9996** | validation set (pooled) |
| Local-only (per region) | **0.985 – 0.996** | validation set (per client) |
| Standard FL (FedAvg) | **0.21 – 0.41** (mean 0.337) | test set (per client) |
| Proposed "FTL" | **0.25 – 0.51** (mean 0.335) | test set (per client) |

— are produced by a pipeline with three fatal problems, any one of which invalidates the conclusions:

1. **The data is essentially fake.** The archived "full run" that produced
   `results (7).csv` / `metrics_report.md` was executed with **all five clients on
   synthetic random-noise data** (see §4, proof in §7). The synthetic generator
   *encodes the class label directly into the red channel*, so the ~99.9% "centralized"
   and ~99% "local" numbers reflect a model reading a leaked label, not classifying
   satellite imagery. The best case ever achieved on real data was a *single* client
   (Europe) with only **5 of 10 EuroSAT classes** populated.

2. **The comparison is apples-to-oranges.** Baselines (centralized, local-only) are
   scored on the **validation** split — the same split used for early-stopping model
   selection — while the federated regimes are scored on the **test** split, and the
   train/val/test partitions are generated with **different random seeds per regime**
   (§8, bugs B7–B9). The 99% vs 30% "gap" is therefore not a measurement of federated
   degradation; it is partly an artifact of measuring different models on different data.

3. **The federated collapse is reproducible even when all clients are identically
   distributed** (§7). This proves the collapse is a *FedAvg implementation defect*
   (weight-space averaging of over-locally-trained, Adam-optimized ResNet-50s, plus a
   round-level metric that measures locally-refit models and hides the failure), **not**
   the "cross-domain distribution shift" story told in the report and
   `Methodology_Implementation_Results_Conclusion.md`.

The engineering scaffolding (class structure, four-regime harness, logging, EDA
notebook, config management) is reasonable and reusable. **The science underneath it is
not.** Details follow.

---

## 1. What the project intended to be

From [README.md](README.md), the report [ppt/Final Draft - Capstone.pdf](ppt/Final%20Draft%20-%20Capstone.pdf),
and the planning doc [cursor-plan/cross-domain_fl_notebook_c134b1e0.plan.md](cursor-plan/cross-domain_fl_notebook_c134b1e0.plan.md):

- **Goal:** Multi-domain satellite land-use/land-cover (LULC) classification under a
  **federated learning** constraint — each geographic region (Africa, Asia, Europe,
  North America, South America) is a client that never shares raw data.
- **Claimed datasets:** EuroSAT (RGB/multispectral) + a BigEarthNet subset (Sentinel-2).
- **Claimed method:** ImageNet-pretrained ResNet-50 backbone + **FedAvg** aggregation +
  a **lightweight domain-adaptation adapter** (per-client BatchNorm or CORAL) that is
  *personalized* (not aggregated) — marketed as "Cross-Domain Federated Transfer Learning."
- **Claimed evaluation:** four regimes (centralized upper bound, local-only lower bound,
  standard FL, proposed FTL) + cross-domain generalization + communication efficiency.
- **Claimed compliance:** "Flower (FLWR)" as the FL framework; multispectral ingestion.

This is a sound and publishable *problem framing*. The gap is entirely in execution.

---

## 2. Artifact inventory (what actually exists in the repo)

| Artifact | Location | Status / role |
|---|---|---|
| Main implementation notebook | [federated_learning_complete.ipynb](federated_learning_complete.ipynb) (root) & [outputs/federated_learning_complete.ipynb](outputs/federated_learning_complete.ipynb) (executed) | 60 cells. The "executed" copy in `outputs/` holds the real run logs. |
| Earlier scaffold | [notebooks/federated_learning_structure .ipynb](notebooks/federated_learning%20_structure%20.ipynb) | Near-identical predecessor of the complete notebook. |
| Colab add-on (precursor) | [notebooks/colab_addon_colab_progress.ipynb](notebooks/colab_addon_colab_progress.ipynb) | Folded into the "Appendix" of the complete notebook. |
| EDA notebook (executed) | [outputs/dataset_exploration_eda_results.ipynb](outputs/dataset_exploration_eda_results.ipynb) | Data-provenance ground truth — the single most honest artifact. |
| EDA notebook (clean) | [dataset_exploration_eda.ipynb](dataset_exploration_eda.ipynb) | Un-executed twin. |
| Inference demo | [outputs/satellite_classification_demo.ipynb](outputs/satellite_classification_demo.ipynb) | **Never executed** (no outputs). Would load the synthetic-trained checkpoint against real EuroSAT. |
| Best checkpoint | [outputs/best_model.pt](outputs/best_model.pt) (94 MB) | `proposed_ftl` global model from the **all-synthetic** run — of no real-world value. |
| Full-run config | [outputs/config (1).yaml](outputs/config%20(1).yaml) | `colab_full_run`: 15 rounds × 4 local epochs, bs 32, RGB. |
| Smoke config | [outputs/config.yaml](outputs/config.yaml) | `smoke_mode` demo config. |
| Results (full) | [outputs/results (7).csv](outputs/results%20(7).csv) | The "primary" numbers — **reproduced by an all-synthetic run (§7)**. |
| Results (intermediate/smoke) | `outputs/results (6).csv`, `results.csv` | Earlier/partial runs; internally inconsistent with each other. |
| Aggregate stats | `outputs/metrics.json`, `outputs/metrics (1).json` | mean/std/min/max of client test acc for FL regimes. |
| Per-class accuracy | [outputs/per_class_accuracy.csv](outputs/per_class_accuracy.csv) | **From a throwaway 3-epoch probe trained and tested on the same val data (§9)** — not the FL model. |
| Auto report | [outputs/metrics_report.md](outputs/metrics_report.md) | Config snapshot + results table + "Cohen kappa 0.585" (also from the probe). |
| Narrative doc | [outputs/Methodology_Implementation_Results_Conclusion.md](outputs/Methodology_Implementation_Results_Conclusion.md) | Report-ready §5–§8 write-up — **repeats the incorrect domain-shift interpretation**. |
| Formal report / slides | `ppt/*.pdf`, `ppt/*.docx`, `ppt/*.pptx` | Submission documents. |

**Environment note:** every attempt to run the notebooks on the local Windows machine
fails at `import torch` with `OSError [WinError 1114] ... c10.dll initialization
routine failed` (see cell 3 output in the executed notebooks). **All results were
produced on Google Colab.** Local reproducibility is currently zero — a broken PyTorch
install must be fixed before the rebuild.

---

## 3. Architecture as built (module map)

The notebook is cleanly layered, and this structure is worth keeping:

```
Config      ExperimentConfig (dataclass), ConfigManager (YAML), ReproducibilityManager
Data        SatelliteDataset(ABC) → EuroSATDataset, BigEarthNetSubsetDataset,
            SyntheticRegionalDataset; DataPreprocessor; GeographicPartitioner;
            stratified_split_indices; SubsetWithTransform
Model       FeatureExtractor(ABC) → ResNetBackbone(resnet50, ImageNet, 3-ch);
            DomainAlignmentLayer(ABC) → IdentityDomainAdapter, BatchNormAdapter, CORALAdapter;
            ClassificationHead (Linear 2048→10); FederatedModel(backbone+adapter+head)
Training    LocalTrainer (train/val/early-stop), FederatedTrainer (local_update/apply_global)
FL core     Client, Server (broadcast→local→aggregate→set), Aggregator(ABC) →
            FedAvgAggregator, DomainAwareAggregator; CommunicationManager/Analyzer;
            run_flower_fedavg_simulation (stub)
Eval        MetricsCalculator, CrossDomainEvaluator, Visualizer
Orchestr.   ExperimentTracker, FederatedExperiment.run_benchmarks() (4 regimes)
Appendix    logging tee, 60-epoch overrides, report/figure/confusion/kappa generation
```

`FederatedModel.get_shared_parameters()` / `set_shared_parameters()` deliberately expose
**only** `backbone.*` and `classifier.*` `named_parameters()`; the domain adapter is kept
per-client (correct design intent). Two consequences the code does **not** account for:

- **BatchNorm running statistics are never shared or aggregated.** `named_parameters()`
  returns BN affine `weight`/`bias` but not the `running_mean`/`running_var` buffers.
  So FedAvg averages BN scale/shift across domains while every client silently keeps its
  own running stats. This is the "FedBN" regime by accident, not by design, and it is
  never analyzed.
- The saved global checkpoint's BN buffers are whatever happened to be in the global
  model object, decoupled from the aggregated affine weights it was paired with.

---

## 4. Dataset analysis — the central failure

### 4.1 The synthetic fallback silently replaces missing data

[federated_learning_complete.ipynb](federated_learning_complete.ipynb) cell 39,
`FederatedExperiment._datasets()`: for each region it tries `EuroSATDataset(root)`
(or `BigEarthNetSubsetDataset` for North America), and **if the folder yields zero
images it substitutes `SyntheticRegionalDataset(n=3200)`** with no error, no warning,
no run-level flag other than a printed `n=3200`.

`SyntheticRegionalDataset._load_image` (cell 10):

```python
x = self._rng.random((64, 64, 3), dtype=np.float32)  # pure noise
x[:, :, 0] += 0.1 * float(lab)                        # <-- LABEL LEAKED INTO RED CHANNEL
return np.clip(x, 0, 1)
```

The label is written into the mean of the red channel. A trivial model learns
"redder ⇒ higher class index" and scores ~100%. **This is why centralized/local hit
0.9996/0.99 — it is label leakage on noise, not classification.** Additionally the
per-sample image is regenerated from a shared advancing RNG on every `__getitem__`, so
"train" and "test" are both drawn from the same trivial generative rule (the rule, not
the pixels, is what's learned), and images are **not deterministic per index** (a
reproducibility wrinkle, worsened under `num_workers>0`).

### 4.2 What the EDA actually found (ground truth)

From the executed [outputs/dataset_exploration_eda_results.ipynb](outputs/dataset_exploration_eda_results.ipynb),
`run_all` cross-region summary (Colab, Drive-backed, `smoke_mode=False`):

| Client (region) | Root | `synthetic` | n_samples | Classes present |
|---|---|:--:|--:|---|
| Africa | data/africa | **Yes** | 3200 | all 10 (random) |
| Asia | data/asia | **Yes** | 3200 | all 10 (random) |
| **Europe** | data/eurosat | **No** | **7774** | **only 5** (Forest, Highway, Pasture, River, SeaLake) |
| North America | data/bigearthnet_subset | **Yes** | 3200 | all 10 (random) |
| South America | data/south_america | **Yes** | 3200 | all 10 (random) |

So in the **best-case** session ever recorded, **4 of 5 clients were synthetic** and the
one real client covered **half** the label space. The EuroSAT materializer
(`_materialize_from_huggingface`, EDA cell 6) wrote:

```
{'AnnualCrop': 0, 'Forest': 1787, 'HerbaceousVegetation': 0, 'Highway': 1505,
 'Industrial': 0, 'Pasture': 1195, 'PermanentCrop': 0, 'Residential': 0,
 'River': 1460, 'SeaLake': 1827}
```

Five classes have **zero** images — a label-name/index mismatch between the HF
`blanchon/EuroSAT` split and the hard-coded `_EUROSAT_NAMES` order silently dropped half
the dataset. Even the "real" client is a broken import.

### 4.3 BigEarthNet and "multispectral" were never real

- `bigearthnet_root` was empty in every recorded session → North America always fell back
  to synthetic. **No BigEarthNet data was ever used in any executed run.**
- Every config sets `num_input_channels: 3` and RGB preprocessing. The N-channel ResNet
  stem code (cell 15) exists but is never exercised. **The "multispectral / Sentinel-2"
  claims in the thesis are unsupported by any run.**

### 4.4 Net effect

The project's entire "cross-domain, multi-sensor, privacy-preserving satellite" premise
was, in practice, **one half-populated RGB dataset plus four buckets of labelled noise**
— and the primary reported numbers used **five** buckets of noise (§7).

---

## 5. Federated architecture analysis

- **Aggregation:** `FedAvgAggregator` (cell 28) is a correct sample-weighted mean over
  the shared params. Fine in isolation.
- **`DomainAwareAggregator` (cell 29) is a no-op** — its `aggregate()` just calls FedAvg
  and `compute_domain_similarity()` returns `np.eye(...)`. Despite the name, there is no
  domain-aware aggregation anywhere.
- **"Flower (FLWR)" is never used.** `run_flower_fedavg_simulation` (cell 31) checks if
  `flwr` is importable and returns `{"status": "skipped_full_sim"}`. All runs used the
  custom loop. The thesis's Flower-compliance claim is not backed by code execution.
- **Domain adapter (the "transfer learning" contribution) is thin and inert in practice:**
  - `BatchNormAdapter` = a per-client `BatchNorm1d(2048)` applied to the pooled feature
    vector, selected by `domain_id`. It is *not* aggregated (correct), but it operates on
    a single 2048-d vector after global average pooling — a very shallow adaptation.
  - `CORALAdapter` + `coral_loss` are only active when `adaptation_method == 'coral'`,
    which **no config uses** (all use `batch_norm`). So CORAL is dead code in every run.
  - Worse, the CORAL loss (cell 22, `LocalTrainer.train_epoch`) is computed between the
    **two halves of a single client's mini-batch** (`feat[:m]` vs `feat[m:]`) — i.e., it
    aligns a domain to *itself*. That is not what CORAL does; the implementation is
    conceptually wrong even if it were enabled.
  - Empirically the adapter does nothing useful: standard FL mean = 0.337, proposed FTL
    mean = 0.335. The "contribution" does not beat its own baseline.
- **Communication efficiency claim is undermined by the numbers it produced.** Each round
  serializes the *entire* ResNet-50 with `pickle` and no compression
  (`CommunicationManager.compress_gradients` is a pass-through). Logged cost is
  **~565 MB per round** (5 uploads + 1 broadcast of ~94 MB each), identical every round.
  There is no efficiency story here — it is the opposite of efficient.

---

## 6. Training-flow analysis

`Server.train_round` (cell 26) per round: broadcast global shared params → each client
trains `local_epochs` (with local early stopping and best-state restore) → FedAvg →
`set_shared_parameters` on the global model → **validate each client's *local* model** →
log `mean_val_acc`.

Two structural issues:

1. **The round metric measures the wrong model.** `mean_val_acc` is computed on each
   client's *locally re-fit* model (right after 4 local epochs), **before** the next
   broadcast. On an easy task the client re-fits in a couple of epochs, so the round
   curve climbs to 0.92→1.00 (see executed logs) **even though the aggregated global
   model is poor.** FL early-stopping (`fl_val_patience`) then selects the "best" round
   by this misleading signal. The genuine global-model quality is only revealed at the
   very end in `evaluate_results` — where it collapses.
2. **Optimizer choice fights FedAvg.** Clients use **Adam** (`lr=1e-3`) with 4 local
   epochs/round. Adam's per-parameter moments are not part of the shared state and are
   effectively stale after every broadcast; combined with many local steps this produces
   large client drift, so averaging the resulting weights lands in a bad region of weight
   space. Standard FedAvg practice (SGD, fewer local steps, LR schedule, or a proximal
   term / FedBN-style handling) is absent.

`run_centralized`, `run_local_only`, and `run_federated_regime` also do **not** share a
compute budget or a data split (see §8), so the four "comparable" regimes are not
actually comparable.

---

## 7. The decisive diagnosis — reproducible collapse on identical data

**Claim:** the FL collapse is an implementation defect, not domain shift.

**Proof (from the executed [outputs/federated_learning_complete.ipynb](outputs/federated_learning_complete.ipynb)):**
the "full run" logs open with

```
=== Geographic distribution ===
  client 0: africa n=3200
  client 1: asia n=3200
  client 2: europe n=3200          <-- europe is ALSO synthetic here (n=3200, not 7774)
  client 3: north_america n=3200
  client 4: south_america n=3200
```

i.e., **all five clients are synthetic**, all drawn from the *same* generative rule
(only the noise seed differs per region). Yet the final per-region test accuracies from
that exact run are:

```
standard_fl:  africa 0.3896  asia 0.4104  europe 0.3833  north_america 0.2104  south_america 0.2917
proposed_ftl: africa 0.2813  asia 0.5146  europe 0.2875  north_america 0.3438  south_america 0.2500
centralized 0.9996 ; local_only 0.985–0.996
```

These are **identical, to the digit, to [outputs/results (7).csv](outputs/results%20(7).csv)
and [outputs/metrics_report.md](outputs/metrics_report.md)** — the numbers the thesis
reports as its primary result.

**Interpretation.** If the collapse were caused by "real vs synthetic domain shift" (the
report's thesis), it could not happen here, because *all clients share one distribution*.
FedAvg on truly IID clients should match the centralized model. Instead it collapses from
~0.99 to ~0.33. Therefore the cause is mechanical:

- weight-space averaging of Adam-trained, over-locally-trained ResNet-50s (client drift),
- with BN running-statistic handling that is never controlled or reported,
- and a round metric that hides the failure until the final test.

The mean FL test accuracy (~0.33 over 10 classes) sits only ~3× chance — the aggregated
global model barely learned the task that a single client learns to 99% in a few epochs.

**Corollary — the reported narrative is wrong.** Both
[outputs/Methodology_Implementation_Results_Conclusion.md](outputs/Methodology_Implementation_Results_Conclusion.md)
and the formal report attribute the gap to "one real Europe client vs four synthetic
clients ⇒ severe domain shift under aggregation." The archived numbers they cite come
from an **all-synthetic** session where that explanation is impossible. The two documents
conflate the real-Europe **EDA** session with the all-synthetic **training** session.

---

## 8. Bug & validity catalogue

Severity: 🔴 invalidates results · 🟠 serious · 🟡 minor/hygiene.

| # | Sev | Location | Defect |
|---|:--:|---|---|
| B1 | 🔴 | `SyntheticRegionalDataset._load_image`, cell 10 | Label leaked into red channel ⇒ trivially separable "task"; ~99.9% "accuracy" is meaningless. |
| B2 | 🔴 | `_datasets()`, cell 39 | Silent synthetic fallback on empty roots; 4–5 of 5 clients were noise in every run, with no run-level provenance flag. |
| B3 | 🔴 | EDA cell 6 materializer | HF EuroSAT label/index mismatch ⇒ 5 of 10 classes written with **0** images; the one "real" client is half-empty and class-skewed. |
| B4 | 🔴 | `Server.train_round`, cell 26 | Round `mean_val_acc` measures locally re-fit client models, not the aggregated global model ⇒ masks aggregation failure and misguides FL early stopping. |
| B5 | 🔴 | Optimizer + local-epoch design, cells 25/26 | Adam + 4 local epochs/round + no proximal/LR schedule ⇒ client drift; FedAvg collapses even on IID data (§7). |
| B6 | 🟠 | `get/set_shared_parameters`, cell 20 | BN `running_mean/var` buffers excluded from aggregation and never synchronized or reported ("accidental FedBN"). |
| B7 | 🔴 | `run_local_only` cell 39; `run_centralized` cell 39 | Baselines report accuracy on the **validation** loader (`test_on_region(cl.val_loader,...)`), i.e. the split used for early-stop selection — optimistic and not the test set. |
| B8 | 🔴 | regime split seeds, cell 39 | Train/val/test built with different seeds per regime (`seed+cid*17` in FL, `seed+cid` in local-only, `seed+rid` in centralized) ⇒ regimes evaluated on **different** partitions; comparison invalid. |
| B9 | 🟠 | `run_benchmarks`, cells 39/47–49 | No shared compute budget: centralized = 60 epochs on 5× pooled data; local-only = 60 epochs/client; FL = 15×4 local passes destroyed by aggregation. Not a controlled comparison. |
| B10 | 🔴 | Appendix G, cell 57 | `per_class_accuracy.csv` and "Cohen kappa 0.585" come from a **fresh probe trained 3 epochs on the pooled val set and evaluated on that same val set** — disconnected from the FL model and train==test. Reported as if diagnostic of the system. |
| B11 | 🟠 | `DomainAwareAggregator`, cell 29 | No-op; identical to FedAvg despite the name and the thesis's "domain-aware" language. |
| B12 | 🟠 | `run_flower_fedavg_simulation`, cell 31 | Flower never runs (stub); "FLWR framework" compliance claim unbacked. |
| B13 | 🟠 | `CORALAdapter.coral_loss` usage, cell 22 | CORAL aligns two halves of the *same* client's batch (self-alignment); also never enabled (`batch_norm` in all configs). Dead, wrong code. |
| B14 | 🟠 | `CommunicationManager.compress_gradients`, cell 30 | Pass-through; full ResNet-50 pickled every round (~565 MB/round). "Communication efficiency" contribution is absent. |
| B15 | 🟡 | Every config | `num_input_channels=3`, RGB only; multispectral/BigEarthNet claims never executed. |
| B16 | 🟡 | cell 15 | ImageNet ResNet-50 fed 64×64 inputs (trained for 224×224); works via adaptive pool but is suboptimal and never ablated. |
| B17 | 🟡 | local env | `torch` fails to load on the local Windows box (c10.dll) — notebooks only ever ran on Colab; no local reproducibility. |
| B18 | 🟡 | `metrics_report.md` | Federated round-history tables render "(no rows)" — the round logging hook wasn't wired for the exported run, so no convergence/communication curves exist for the report. |
| B19 | 🟡 | `results (6/7).csv`, `results.csv` | Three mutually inconsistent "results" files with different centralized accuracies (0.50 / 0.57 / 0.9996); no commit/data hash ties any to a config. |

---

## 9. Metrics & evaluation integrity

- The single most-cited "quality" numbers in the report — **per-class accuracy** and
  **Cohen's kappa (0.585)** — are from the throwaway probe (B10), not the federated model.
  The narrative built on them ("Industrial/River/Residential are visually ambiguous") is
  unfounded: those classes were largely **absent or synthetic**, so their "accuracy" is
  noise.
- Macro-F1 is computed correctly (sklearn) and logged per region, which is good practice
  to keep — but it inherits all the data/split problems above.
- No confidence intervals, no multiple seeds, no significance testing. `metrics.json`
  reports a std across 5 clients, but with non-comparable splits it is not interpretable.

---

## 10. Documentation & claims integrity

The written record oversells and, in places, misreports:

- **Report + methodology doc** attribute the FL collapse to real-vs-synthetic domain
  shift, using numbers that came from an **all-synthetic** run (§7). This is a factual
  error, not just an over-interpretation.
- **To its partial credit**, the formal report *does* disclose (in §5/§6/limitations)
  that "four out of five clients use synthetic data" and that this is a threat to
  validity. That honesty is good and must be preserved — but the results tables and
  headline claims are not consistent with that disclosure.
- "Flower framework," "multispectral," "BigEarthNet," "communication efficiency," and
  "domain-aware aggregation" all appear as contributions but have **no executed code path**
  behind them.

For a conference submission, a reviewer running the notebook once would surface B1/B2
immediately. This is the top risk to address.

---

## 11. Real-life implications

- **As a proof-of-concept for federated satellite classification, it currently proves
  nothing.** A model that reads a leaked label from noise, aggregated by a FedAvg loop
  that collapses on IID data, tells us nothing about privacy-preserving cross-region LULC.
- **The privacy/federation motivation is real and valuable** (regions/agencies can't
  share raw imagery; satellite data is heterogeneous across sensors and geographies) — the
  *framing* is publishable; only the *execution* is not.
- **A corrected version is very achievable.** EuroSAT (real, all 10 classes) under FedAvg
  reaches ~90%+ in the literature; the non-IID and cross-sensor story (EuroSAT ↔
  BigEarthNet, or per-region partitions of a real global dataset) is a legitimate research
  contribution if measured honestly.

---

## 12. What is salvageable (keep in the rebuild)

- The **layered class design** and the **four-regime benchmark harness** shape
  (centralized / local-only / standard-FL / proposed) — conceptually right.
- **Config management** (`ExperimentConfig` + YAML round-trip) and **ExperimentTracker**.
- The **EDA notebook's scanners and provenance reporting** — it is the artifact that
  actually told the truth; extend it into a hard *gate* (fail the run if any client is
  synthetic or any class is empty).
- **Macro-F1 + confusion-matrix + per-class** plumbing (once pointed at the real model).
- The **domain-adapter interface** (`DomainAlignmentLayer`) as a place to implement a
  *real* method (FedBN, CORAL done correctly, or FedProx) — but the current bodies must be
  rewritten.

---

## 13. Recommendations for the top-down rebuild

Ordered by leverage. Treat items 1–4 as non-negotiable gates before any result is quoted.

1. **Use only real data, and fail loudly otherwise.** Delete the silent synthetic
   fallback (or gate it behind an explicit `allow_synthetic=False` that raises). Materialize
   **all 10 EuroSAT classes** correctly (verify the HF label→name mapping; assert
   per-class counts > 0). Add a real second domain (a genuine BigEarthNet RGB/MS subset, or
   partition a single global dataset by region) so "cross-domain" is real.
2. **Fix the evaluation protocol.** One fixed stratified train/val/test split per client,
   generated once with a single seed and reused by **all** regimes. Report **test**
   accuracy for every regime; use val only for early stopping. Never report the selection
   metric as a result.
3. **Fix FedAvg.** Switch clients to SGD (or FedAdam at the server), reduce local epochs
   (1–2), add an LR schedule, and evaluate the **aggregated global model** each round (not
   locally-refit clients). Establish that FedAvg matches centralized on IID data before
   introducing heterogeneity — this is the sanity check the current code fails.
4. **Control BatchNorm explicitly.** Choose and document a policy: aggregate BN buffers,
   or adopt FedBN deliberately, and ablate it. Stop leaving it to accident.
5. **Make the "proposed" method a real contribution.** Either implement CORAL correctly
   (source vs target domain alignment) or commit to FedBN/FedProx/personalization, and
   show it beats standard FedAvg on **real** heterogeneous data with CIs over ≥3 seeds.
6. **Back every claim with a run.** If the paper says Flower, run Flower (and check parity
   with the custom loop). If it says multispectral, run ≥4-band inputs. If it says
   communication-efficient, implement compression and report accuracy-per-MB honestly.
7. **Reproducibility.** Fix the local PyTorch/CUDA install (B17), pin versions, log data
   hashes + git commit per run, and consolidate to a single canonical results file.
8. **Rewrite the narrative from the corrected numbers**, and drop the domain-shift
   explanation for the synthetic collapse.

---

## 14. One-paragraph summary for the record

The existing capstone builds a competent-looking federated-learning scaffold around a
non-existent experiment: four (often five) of five clients are synthetic noise with the
label leaked into the pixels, the one real client is a half-broken EuroSAT import, the
baselines and federated regimes are scored on different splits with different seeds, the
"convergence" metric measures the wrong model, the headline per-class/kappa numbers come
from a throwaway probe, and the reported "domain-shift" story is contradicted by the fact
that the exact published numbers reproduce on all-identical synthetic clients — proving
the ~99%→~33% collapse is a FedAvg implementation defect, not science. The framing and
code structure are reusable; the data, the experimental protocol, the aggregation loop,
and the results are not. **Rebuild required.**
