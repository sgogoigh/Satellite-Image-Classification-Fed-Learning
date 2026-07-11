# Rebuild Plan — Cross-Region Federated Learning for Satellite Image Classification

> **Status:** design document for a top-down rebuild.
> **Companion:** read [previous-progress.md](previous-progress.md) first — it is the audit of the
> old pipeline and defines the 19 defects (B1–B19) that this plan must close.
> **Author stance:** build the smallest thing that is *honest, reproducible, and comparable*,
> then extend. Every reported number must survive a reviewer running the repo once.
> **Date:** 2026-07-11.

---

## 0. TL;DR — the decisions at a glance

| Axis | Old (broken) | New (this plan) |
|---|---|---|
| **Data** | 4–5 of 5 clients = synthetic noise with the label leaked into the red channel | **100% real EuroSAT** (all 10 classes), no synthetic fallback — pipeline hard-fails if data is missing |
| **"Regions"** | Empty per-continent folders → synthetic | **Emulated** by partitioning one real dataset across K clients (no continent data collected) |
| **Division technique** | one-folder-per-region, silent fallback | **Dirichlet(α) label-skew partitioning** (+ optional per-client sensor-shift simulation) via `flwr-datasets`; indices saved to disk and reused by every regime |
| **Cross-domain story** | claimed, never real | **Real feature shift**: simulated sensor corruptions (primary) and/or a **multi-dataset** track (EuroSAT+AID+NWPU+UC-Merced under a shared taxonomy) for genuine cross-sensor domain shift |
| **FL framework** | custom loop; "Flower" was a stub | **Flower (`flwr`) Simulation Engine** for real, citable FedAvg/FedProx/FedAdam |
| **Aggregation** | Adam clients, 4 local epochs → weight-space collapse | **SGD clients, 1–2 local epochs, LR schedule**; server-side FedAdam optional; verified against centralized on IID |
| **Proposed method** | inert BN/CORAL adapter, no gain | **Personalized Federated Transfer Learning** = pretrained backbone + **FedBN** (local BN, fixes feature shift) + FedProx term + optional personalized head |
| **Eval** | baselines on *val*, FL on *test*, different seeds per regime; probe-based per-class/kappa | **one fixed train/val/test split per client**, test-only reporting, **global model evaluated every round**, ≥5 seeds with 95% CI |
| **Comm efficiency** | 565 MB/round, no compression, claimed anyway | ResNet-18; **real** MB/round logged; accuracy-per-MB; compression as an ablation |
| **Reproducibility** | Colab-only, torch broken locally, 3 inconsistent result files | pinned env, seeded, config-driven, one canonical results store, data + git hash per run |

**Recommended primary experiment:** EuroSAT-RGB (all 10 classes) → Dirichlet-partitioned into K regional clients → FedAvg / FedProx baselines vs a **FedBN-based personalized FTL** proposed method → evaluated with a fixed protocol, an α-heterogeneity sweep, a client-scale sweep, and a **leave-one-client-out** generalization test. Multispectral (13-band) and multi-dataset cross-domain are scoped as clearly-labelled extensions.

---

## 1. Purpose, scope, and non-goals

**Purpose.** Produce a reproducible proof-of-concept that federated learning can train a shared
satellite land-use/land-cover (LULC) classifier across many *regional* nodes that never share raw
imagery, and that a lightweight transfer-/domain-adaptation method improves over vanilla FedAvg
under realistic heterogeneity.

**Scope (what we WILL do).**
- Use **real** benchmark satellite imagery.
- **Emulate** cross-region/cross-sensor federation by *partitioning* and *controlled shift*, not by
  collecting continent-specific data (per the sponsor's explicit instruction).
- Implement standard, citable FL (Flower) with correct evaluation.
- Deliver an honest four-plus-regime comparison with statistics and ablations.

**Non-goals (explicitly out of scope, stated so we never claim them).**
- No real per-continent data acquisition.
- No formal differential-privacy guarantee (we provide *data minimization*, and scope DP-SGD as
  future work — we will not claim "privacy-preserving" without qualification).
- No pixel-level segmentation / thematic maps (this is patch classification).
- No production deployment; this is a research POC.

---

## 2. Design principles (the three rules every experiment obeys)

1. **Honesty.** Only claim what an executed code path produced. No silent fallbacks, no simulated
   data masquerading as real, no metric computed on a different model than the one under test.
2. **Comparability.** All regimes share the *same* data splits, the *same* per-client compute
   budget, the *same* backbone/init protocol, and the *same* seeds. The only thing that changes is
   the variable under study.
3. **Reproducibility.** A fresh clone + `pip install` + one command reproduces every number
   (±seed noise). Partition indices, configs, data hashes, and git commit are recorded per run.

---

## 3. Dataset decision and rationale

### 3.1 Primary dataset — EuroSAT (RGB), all 10 classes

**Choice:** [EuroSAT](https://github.com/phelber/eurosat) — 27,000 Sentinel-2 patches, 64×64,
10 LULC classes, geo-referenced. Available RGB (`blanchon/EuroSAT_RGB`) and 13-band multispectral
(`blanchon/EuroSAT_MSI`) on the Hugging Face Hub; also `torchvision.datasets.EuroSAT`.

**Why EuroSAT is the right backbone for an emulated cross-region study:**
- **Real, clean, standard.** It is the canonical FL/vision satellite benchmark, so our results are
  directly comparable to published FedAvg/FedProx/FedBN numbers (IID FedAvg ≈ 0.95; Dirichlet
  non-IID ≈ 0.90 ± 0.04 in the literature). This gives a *sanity yardstick* the old project lacked.
- **10 real classes** → fixes the "5 empty classes" defect (B3) outright, provided we load labels
  from the dataset's own `ClassLabel` feature (never a hard-coded name list).
- **Small (64×64)** → a 5–50 client Flower simulation runs on a single Colab/consumer GPU.
- **Class frequencies vary naturally by geography** (a coast has SeaLake/River, an arid region has
  more AnnualCrop/Pasture), so a *label-distribution* partition is a faithful analog of
  "different continents see different land cover."

**Loader contract (kills B1, B2, B3):**
- Load via `flwr-datasets` `FederatedDataset(dataset="...EuroSAT_RGB...")` or torchvision.
- Read class names from the dataset's `ClassLabel.names` — never assume an order.
- **Data-integrity gate** (a hard assertion at load): exactly 10 classes present; every class count
  > 0; total ≈ 27,000; record a SHA-256 of the sorted file/label manifest. **If any assertion
  fails, raise — no synthetic fallback exists in the codebase at all.**

### 3.2 Emulating "regions" and "scale" without collecting regional data

The sponsor wants to *demonstrate feasibility at continent scale*, not gather continent data. We
achieve this three ways, all reproducible:

- **Label-distribution heterogeneity (primary).** Dirichlet(α) partition of EuroSAT into K clients
  (§4). Each client is a "region/agency" with a distinct class prior. Sweeping α shows the method
  across the IID→severe-non-IID spectrum.
- **Scale demonstration.** Run K ∈ {5, 10, 20, 50} clients with per-round client **subsampling**
  (`fraction_fit < 1`). This proves the *architecture* scales to many regional nodes and partial
  participation — the "continent scale is possible" claim — using only EuroSAT.
- **Feature/sensor shift (Track A+, optional but recommended for the transfer-learning story).**
  Apply a **fixed, seeded, per-client photometric/atmospheric transform** (brightness, contrast,
  gamma, hue shift, haze/fog, Gaussian blur, JPEG artifacts, sensor noise) to emulate different
  satellites/atmospheres/seasons. This produces genuine **covariate (feature) shift** on top of
  label shift, which is precisely what motivates FedBN/domain adaptation. It is honest because we
  **declare it as simulated sensor variation** and it is fully reproducible.

### 3.3 Genuine cross-domain track (Track B, advanced/flagship extension)

For a *real* (not simulated) cross-sensor/cross-resolution contribution, make **each client a
different real RS scene dataset** mapped to a **shared coarse taxonomy**:

| Client (domain) | Dataset | Sensor / source | Native classes |
|---|---|---|---|
| Domain 1 | EuroSAT | Sentinel-2, 10 m, 64px | 10 |
| Domain 2 | [AID](https://captain-whu.github.io/AID/) | Google Earth aerial, 600px | 30 |
| Domain 3 | [NWPU-RESISC45](https://gcheng-nwpu.github.io/) | aerial, 256px | 45 |
| Domain 4 | [UC-Merced](http://weegee.vision.ucmerced.edu/datasets/landuse.html) | USGS aerial, 0.3 m, 256px | 21 |
| (held-out) | PatternNet / WHU-RS19 | aerial | 38 / 19 |

Map all to a **shared 6–7 macro-class taxonomy** — e.g. `{Agriculture/Cropland, Forest/Vegetation,
Water, Residential, Industrial/Commercial, Transport/Highway, Barren/Other}` — following the
standard cross-domain RS protocol (pair datasets on their shared classes; document the mapping
table explicitly and report results at both fine and coarse granularity). This is the authentic
version of the old project's claim and enables real **leave-one-domain-out** generalization.

**Recommendation.** Ship **Track A + A+** as the core (reproducible, controllable, low-risk,
publishable on its own). Treat **Track B** as the flagship cross-domain experiment / stretch goal —
it adds real domain shift but carries taxonomy-mapping risk (§16), so it is additive, not blocking.

### 3.4 Multispectral option (makes the "multispectral" claim real, or we drop it)

Provide an MS track using `blanchon/EuroSAT_MSI` (13 bands): expand the backbone's first conv to 13
input channels (copy ImageNet RGB weights into the 3 visible bands, initialize the rest as the mean
of RGB weights — the technique the old code sketched but never ran). Run it as a labelled ablation
(RGB vs MS). **If we do not run it, we delete every "multispectral" claim from the writeup** (fixes
B15). No unsupported claims.

---

## 4. Data partitioning — the division technique (fixes B2, B7, B8)

**Tool:** `flwr-datasets` partitioners (`DirichletPartitioner`, `PathologicalPartitioner`,
`ShardPartitioner`) — reproducible, standard, HF-native.

**Protocol:**
1. Start from EuroSAT's **canonical train/test** if provided; otherwise create a global stratified
   80/20 dev/test split with a fixed seed.
2. **Partition the dev set across K clients with `DirichletPartitioner(alpha=α, partition_by="label")`.**
   α controls skew: `α=100` ≈ IID, `α=0.5` moderate, `α=0.1` severe.
3. **Within each client**, make a **fixed stratified train/val/test split** (e.g. 70/15/15), seed =
   `base_seed` (not `base_seed + regime`). Persist the exact indices to
   `data/partitions/{dataset}_{K}clients_alpha{α}_seed{seed}.json`.
4. **Every regime (centralized, local-only, all FL variants) loads these same saved indices.**
   Centralized = the *union* of client train sets; centralized test = the *union* of client test
   sets. No regime ever regenerates a split. (This single change kills B7 and B8.)
5. For **leave-one-client-out**, hold out client *j* entirely from training; evaluate the final
   global model on client *j*'s test split (its data never seen in training).

**Reported partition diagnostics (part of the EDA gate):** per-client class histograms, a
class×client heatmap, per-client sample counts, and the realized heterogeneity (e.g. average
per-client label entropy) so the α setting is auditable.

---

## 5. Model architecture (fixes B16)

- **Default backbone:** ResNet-18, ImageNet-pretrained (transfer learning). Lighter than ResNet-50
  → feasible many-client simulation; standard in FL benchmarks. ResNet-50 available as a
  capacity ablation.
- **Input handling:** document one resize policy and keep it fixed. Two supported: (a) native 64×64
  with a stride-adjusted stem, or (b) resize to 224 (standard ImageNet resolution) — we default to
  (b) for pretrained fidelity and ablate (a). No silent 64→ResNet50 mismatch.
- **Head:** single linear classifier (num_classes). Optional 2-layer MLP head for the personalized
  variant.
- **Normalization policy is a first-class, documented choice** (fixes B6): the FedAvg baseline
  aggregates BN affine **and** running buffers correctly (buffers included in the shared state, or
  swap BN→GroupNorm for a BN-free control); the **proposed** method uses **FedBN** (BN kept local).
  We ablate {aggregate-BN, FedBN, GroupNorm} head-to-head.
- **MS variant:** 13-channel first conv as in §3.4.

---

## 6. Federated method and the proposed contribution (fixes B5, B6, B11, B12, B13, B14)

### 6.1 Baselines (all via Flower strategies — real, not stubs)

| Regime | Definition | Purpose |
|---|---|---|
| **Centralized** | one model on pooled client train data, matched compute | upper bound |
| **Local-only** | each client trains alone from the same pretrained init | lower bound / no-collaboration |
| **FedAvg** | Flower `FedAvg`, SGD clients | standard FL |
| **FedProx** | Flower `FedProx` (proximal μ) | non-IID-robust baseline |
| **FedAdam** *(opt.)* | server-side adaptive optimizer | strong baseline |
| **SCAFFOLD** *(opt.)* | control-variate correction | strong non-IID baseline |

### 6.2 Proposed — "Personalized Federated Transfer Learning" (PFTL)

A composition of well-founded, individually-ablated components, targeting the exact failure modes
the audit exposed:

1. **Transfer learning:** ImageNet-pretrained backbone, shared and federated (the "TL" in FTL).
2. **FedBN (core):** batch-norm layers are **kept local** and never aggregated — directly handles
   the cross-sensor/atmospheric **feature shift** (Track A+ / B) and is the principled fix for the
   BatchNorm defect (B6). Chosen because FedBN provably beats FedAvg and FedProx under feature-shift
   non-IID and has an official Flower baseline.
3. **FedProx stabilization:** proximal term to limit client drift under label skew.
4. **Optional personalized head:** each client keeps a local classifier head (shared backbone) — a
   simple, strong personalization for label skew; reported separately so its contribution is clear.
5. **Optional correct CORAL/feature alignment:** if included, align each client's feature covariance
   to a shared reference broadcast by the server (not two halves of one batch as before, B13);
   otherwise dropped. We will not ship a mislabeled or inert adapter.

**Claim discipline:** the "proposed method" wins only if it beats FedAvg/FedProx on **real** test
data with non-overlapping CIs over ≥5 seeds. If it doesn't, we report that honestly (a negative or
mixed result is publishable; a fabricated positive is not).

### 6.3 Communication (fixes B14)

Log **actual** bytes/round from Flower's message sizes. Report total MB, MB-to-target-accuracy, and
accuracy-per-MB. Add optional **compression** (top-k sparsification or 8-bit quantization) as an
ablation and report the accuracy/communication trade-off. Never claim efficiency without the curve.

---

## 7. Corrected end-to-end training flow (fixes B4, B5, B9, B10)

The canonical run, identical across regimes except the variable under study:

```
[0] Integrity gate:  assert real data, 10 classes, counts>0, record data hash + git commit
[1] Partition:       load saved Dirichlet indices for (dataset, K, α, seed); build per-client
                     train/val/test; centralized = unions of those exact splits
[2] Init:            same pretrained backbone; seed model init; log seed
[3] Per regime:
      centralized ->  train on pooled train, early-stop on pooled val, evaluate on pooled test
      local-only  ->  per client: train on its train, early-stop on its val, test on its test
      FL (Flower) ->  for each round r in 1..R:
                        server samples fraction_fit clients
                        broadcast global params
                        each client: SGD for E local epochs (E in {1,2})
                        server AGGREGATES (FedAvg/Prox/BN policy)
                        >>> EVALUATE THE GLOBAL MODEL <<<  (B4 fix)
                          - centralized eval: global model on a held-out global test set
                          - federated eval:   weighted mean of client test metrics
                        log round metrics + comm bytes; early-stop on GLOBAL val
[4] Final eval:      the SAME final model is used for ALL diagnostics (B10 fix):
                     overall acc, macro-F1, per-class F1, confusion matrix, kappa — on TEST
[5] Persist:         one canonical results record (config, seed, hashes, all metrics, curves)
```

**Compute-budget parity (B9):** define a single budget — *total local gradient steps per client* —
and hold it constant across regimes (centralized gets the client-count-scaled equivalent). Document
the arithmetic in the config so "centralized 60 epochs vs FL 15×4" mismatches can't recur.

**Optimizer (B5):** clients use **SGD + momentum**, `E∈{1,2}` local epochs, cosine/step LR schedule,
gradient clipping. Server may use FedAdam. This is the configuration under which FedAvg is known to
be stable — and we *verify* it (sanity gate G4).

---

## 8. Evaluation protocol (fixes B4, B7, B8, B10)

- **Splits:** one fixed per-client train/val/test, reused everywhere (§4). **Test-only reporting;
  val is for early stopping only.**
- **Model under test:** always the actual global/final model; diagnostics never come from a
  throwaway probe.
- **Metrics (all on test):**
  - Global: overall accuracy, **macro-F1**, per-class F1/precision/recall, confusion matrix,
    balanced accuracy, Cohen's κ.
  - Per-client: mean ± std accuracy, and **worst-client accuracy** (fairness under heterogeneity).
  - Convergence: accuracy vs round; **rounds-to-target** (e.g. rounds to reach 85%).
  - Communication: total MB; MB-to-target; accuracy-per-MB.
  - **Generalization gaps:** centralized − federated; **participating − held-out (LOCO)**.
- **Statistics:** ≥5 seeds; report **mean ± 95% CI**; paired **Wilcoxon/t-test** for
  proposed-vs-baseline; no single-run claims.
- **Cross-domain generalization:** leave-one-client-out (Track A) and leave-one-domain-out (Track B)
  — train on the rest, evaluate the global model on the unseen client/domain test set.

---

## 9. Experiment matrix (what actually runs)

| Study | Variable | Settings | Regimes | Seeds |
|---|---|---|---|---|
| E1 Sanity | IID check | α=100, K=5 | centralized, FedAvg | 3 |
| E2 Main | heterogeneity | α ∈ {100, 1.0, 0.5, 0.1}, K=10 | centralized, local, FedAvg, FedProx, **PFTL** | 5 |
| E3 Scale | #clients + participation | K ∈ {5,10,20,50}, fraction_fit ∈ {0.2,0.5,1.0} | FedAvg, **PFTL** | 3 |
| E4 BN policy | normalization | aggregate-BN / FedBN / GroupNorm | FedAvg vs **PFTL** | 5 |
| E5 Sensor shift | feature shift on/off | Track A vs A+ | FedAvg, FedProx, **PFTL** | 5 |
| E6 Modality | RGB vs MS(13-band) | EuroSAT_RGB vs _MSI | centralized, FedAvg, **PFTL** | 3 |
| E7 LOCO | unseen client/domain | leave-one-out | FedAvg, FedProx, **PFTL** | 5 |
| E8 Comm | compression | none / top-k / 8-bit | FedAvg, **PFTL** | 3 |
| E9 Cross-domain* | multi-dataset | Track B (EuroSAT+AID+NWPU+UCM) | local, FedAvg, **PFTL** | 3 |

\*E9 depends on Track B taxonomy work; ships if time permits.

**Exit criterion for credibility:** E1 must show FedAvg ≈ centralized under IID. If it does not,
stop and fix the FL loop before running anything else. (This is the check the old project failed.)

---

## 10. Framework and tech stack

- **FL:** [Flower](https://flower.ai/) (`flwr`) Simulation Engine — `ClientApp` / `ServerApp` /
  `run_simulation`, built-in `FedAvg`/`FedProx`/`FedAdam` strategies, Ray-backed virtual clients on
  one GPU. Custom strategy subclass only for FedBN/PFTL.
- **Partitioning/data:** [`flwr-datasets`](https://flower.ai/docs/datasets/) `DirichletPartitioner`
  over HF EuroSAT.
- **DL:** PyTorch + torchvision (pretrained ResNet); torchmetrics for metrics.
- **Config:** Hydra or plain YAML dataclasses (typed), one config per experiment.
- **Tracking:** Weights & Biases *or* MLflow; always also write local `results.jsonl` + `config.yaml`
  + `metrics.csv` so nothing depends on a cloud account.
- **Env:** pinned `requirements.txt` / `pyproject.toml` + a working CUDA build (fixes B17); a
  `conda`/`venv` recipe; Colab notebook that installs the pinned set. Verify `import torch` on the
  target machine as step zero.

---

## 11. Repository structure and engineering

Move logic out of monolithic notebooks into a small tested package; notebooks become thin drivers.

```
src/fedsat/
  data/        loaders.py (EuroSAT/AID/... + integrity gate), partition.py (Dirichlet, save/load),
               shift.py (per-client sensor corruptions), taxonomy.py (Track B mapping)
  models/      backbones.py (resnet18/50, MS stem), heads.py, norm.py (BN/GN/FedBN policy)
  fl/          client_app.py, server_app.py, strategies.py (FedBN/PFTL), task.py
  train/       centralized.py, local_only.py, loops.py (shared budget), earlystop.py
  eval/        metrics.py (torchmetrics), loco.py, communication.py, stats.py (CIs, tests)
  utils/       seeding.py, hashing.py, config.py, tracking.py
configs/       e1_sanity.yaml ... e9_crossdomain.yaml, base.yaml
notebooks/     00_eda_gate.ipynb, 01_run_experiment.ipynb, 02_analysis_figures.ipynb
tests/         test_partition_disjoint.py, test_integrity_gate.py, test_overfit_one_batch.py,
               test_fedavg_equiv_centralized_iid.py, test_reproducibility.py
results/       <run_id>/{config.yaml, metrics.csv, curves.json, confusion.png, data_hash.txt}
data/partitions/  saved index files (committed or regenerated deterministically)
README.md, requirements.txt, pyproject.toml, LICENSE
```

---

## 12. Reproducibility and sanity gates (the checks the old project lacked)

Each gate is an automated test that must pass before results are trusted:

| Gate | Assertion |
|---|---|
| **G1 Integrity** | real data only; 10 classes; every class count > 0; data hash recorded; **no synthetic code path exists** |
| **G2 Disjoint splits** | per-client train/val/test index intersections are empty; identical splits loaded by every regime; held-out client truly unseen |
| **G3 Overfit-one-batch** | model reaches ~100% on a single batch → learning code is correct |
| **G4 FedAvg≡Centralized (IID)** | under α=100, FedAvg reaches within ~1–2% of centralized → aggregation loop is correct (the decisive check) |
| **G5 Global-eval** | round metrics come from the aggregated global model, not locally-refit clients |
| **G6 Determinism** | same seed + config → identical metrics (within float tolerance) |
| **G7 Budget parity** | logged gradient-steps/epochs equal across regimes (within the defined budget) |
| **G8 No-leak diagnostics** | per-class/confusion/κ computed from the final test model, from held-out test data |

---

## 13. Phased execution plan

Each phase has a deliverable and an exit criterion; later phases don't start until the gate passes.

- **P0 — Environment & data (deliverable: green G1, fixed env).** Working CUDA PyTorch locally +
  Colab; EuroSAT loaded via HF/torchvision with correct labels; integrity gate + EDA notebook with
  per-class counts and hashes. *Exit:* G1 passes; all 10 classes non-empty; hash logged.
- **P1 — Centralized baseline (deliverable: ~95%+ EuroSAT).** Pretrained ResNet-18, fixed splits,
  full metric suite. *Exit:* matches literature ballpark; G3 passes.
- **P2 — FL core in Flower (deliverable: green G4).** FedAvg via Simulation Engine on the IID
  partition; global-model round evaluation; comm logging. *Exit:* **G4 (FedAvg≈centralized on IID)**
  and G5 pass. **This is the make-or-break gate.**
- **P3 — Non-IID partitioning (deliverable: E2 α-sweep).** Dirichlet partitions; local-only +
  FedAvg + FedProx across α. *Exit:* expected degradation-with-heterogeneity curve reproduced.
- **P4 — Proposed PFTL (deliverable: E4/E5).** FedBN + FedProx (+ personalized head); BN-policy and
  sensor-shift ablations. *Exit:* component contributions isolated; CIs computed.
- **P5 — Scale & communication (deliverable: E3/E8).** Client-scale and participation sweeps;
  compression trade-off. *Exit:* scale demonstrated; comm curves produced.
- **P6 — Generalization (deliverable: E7, opt. E6/E9).** LOCO on Track A; optional MS and Track B
  cross-domain. *Exit:* unseen-client generalization quantified.
- **P7 — Analysis & writeup (deliverable: figures, tables, paper).** Aggregate over seeds; CIs and
  significance tests; convergence/per-region/comm/confusion figures; rewrite report from the *new*
  numbers; honest limitations. *Exit:* every claim traces to a run in `results/`.

---

## 14. Traceability — every audit defect has a fix (cross-check #1)

| Bug | Defect (from audit) | Fix in this plan | Section |
|---|---|---|---|
| B1 | label leaked into red channel (synthetic) | no synthetic data; real EuroSAT | §3.1 |
| B2 | silent synthetic fallback | fallback removed; hard-fail integrity gate G1 | §3.1, §12 |
| B3 | HF label mismatch → 5/10 classes empty | load labels from `ClassLabel`; assert all 10 non-empty | §3.1, §12 |
| B4 | round metric measured locally-refit models | evaluate the **global** model each round (G5) | §7, §8 |
| B5 | Adam + 4 local epochs → weight collapse | SGD, 1–2 local epochs, LR schedule; verified by G4 | §6, §7 |
| B6 | BN running buffers never handled | explicit BN policy; **FedBN**; ablation | §5, §6.2 |
| B7 | baselines on val, FL on test | test-only reporting; val for early-stop only | §4, §8 |
| B8 | different split seeds per regime | one saved partition reused by all regimes | §4 |
| B9 | no shared compute budget | explicit gradient-step budget, logged (G7) | §7, §12 |
| B10 | per-class/κ from throwaway probe | diagnostics from the final test model (G8) | §7, §8 |
| B11 | `DomainAwareAggregator` no-op | real Flower strategies; no mislabeled aggregators | §6.1 |
| B12 | Flower was a stub | Flower Simulation Engine runs all FL | §6, §10 |
| B13 | CORAL self-aligned batch halves / dead | FedBN core; CORAL only if aligned to broadcast reference | §6.2 |
| B14 | 565 MB/round, no compression, claimed anyway | ResNet-18; real MB logged; compression ablation | §6.3 |
| B15 | multispectral never run | MS track (13-band) or claim deleted | §3.4 |
| B16 | 64px into ResNet-50 | ResNet-18, documented resize policy, input ablation | §5 |
| B17 | torch broken locally; Colab-only | pinned env; verify import; local+Colab parity | §10, §13 P0 |
| B18 | round history not persisted | Flower History → curves persisted per run | §8, §11 |
| B19 | 3 inconsistent result files | one canonical results store + config/hash per run | §7, §11 |

*(Every one of B1–B19 maps to a concrete change. No defect is carried forward.)*

---

## 15. Disposition of the old code

- **Keep (reuse the shape, not the internals):** the four-regime benchmark concept; the
  config-dataclass idea; the EDA scanners (upgrade into the G1/G2 gate); the metrics interface.
- **Rewrite:** the FL loop (→ Flower), the model normalization handling (→ BN policy/FedBN), the
  evaluation protocol (→ fixed splits, global eval, test-only), the data layer (→ real loaders +
  partitioner + integrity gate).
- **Delete outright:** `SyntheticRegionalDataset` and the silent fallback; `DomainAwareAggregator`
  (no-op); the CORAL self-alignment; the throwaway-probe per-class/κ appendix; the three stray
  `results*.csv`; the `run_flower_*` stub.
- **Archive:** `best_model.pt` (synthetic-trained — of no value; keep only as a historical artifact,
  clearly labelled, not for the demo).

---

## 16. Threats to validity and honesty statement (goes in the paper verbatim-ish)

- **Simulated heterogeneity ≠ real regional data.** Dirichlet label skew and seeded sensor
  corruptions *emulate* cross-region/cross-sensor conditions; they are not a substitute for real
  continental data. We state this plainly and position it as a controlled, reproducible study of the
  *mechanism*, with Track B (real multi-dataset) as the closest real-world proxy.
- **EuroSAT is comparatively easy** (high accuracy ceiling). The interesting axes are therefore
  *heterogeneity (α)*, *scale (K)*, *feature shift*, and *unseen-client generalization (LOCO)* —
  not the absolute IID accuracy.
- **Track B taxonomy mapping is subjective.** We publish the full mapping table, report both fine
  and coarse results, and treat any many-to-one merges conservatively.
- **Privacy.** FL here provides data minimization, not a formal guarantee. We scope DP-SGD /
  secure aggregation as future work and do not claim differential privacy.

---

## 17. Real-world positioning

The corrected POC models a realistic scenario: multiple **mapping agencies / satellite operators /
regional data centers** collaboratively train one LULC classifier **without exchanging raw imagery**,
tolerating that each holds a different mix of land-cover classes and slightly different sensors. The
scale and unseen-client experiments argue that the same architecture would extend to many real
regional nodes (including on-orbit/edge settings à la federated learning on space platforms). This
is a defensible, honest contribution — unlike the prior version, which demonstrated only that a
network can read a label leaked into a pixel.

---

## 18. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Flower/Ray simulation setup friction on Windows | med | develop on Colab/Linux; pin versions; document; CPU-small smoke test first |
| G4 fails (FedAvg ≠ centralized on IID) | low if §6/§7 followed | that's the point of G4 — debug the loop before proceeding; do not run downstream studies |
| EuroSAT ceiling too high to separate methods | med | lean on α-sweep, sensor shift, LOCO, and Track B where gaps are larger |
| Track B taxonomy disputes | med | conservative mapping, published table, fine+coarse reporting; keep Track B optional |
| Proposed method doesn't beat baselines | med | report honestly; a rigorous negative/mixed result is still a valid capstone/paper |
| Compute for 5 seeds × full matrix | med | prioritize E1→E2→E4→E7; others as time allows; use small K and ResNet-18 |

---

## 19. Self-review checklist (cross-check #2)

Re-verifying this plan against the audit and the original goals:

- [x] **Every audit defect B1–B19 has an explicit fix** (§14 table — all 19 mapped).
- [x] **No synthetic data anywhere**; missing data hard-fails (G1). — kills the root cause.
- [x] **All regimes share one split and one compute budget**; test-only reporting. — kills the
      apples-to-oranges comparison.
- [x] **FL correctness is *gated* (G4) before any result is believed** — directly targets the
      reproducible-collapse finding; FedAvg must match centralized on IID.
- [x] **Global model is evaluated each round**; diagnostics come from the real test model. — kills
      the masking metric and the probe.
- [x] **The "proposed" method is real (FedBN/FedProx/personalization), ablated, and only claimed if
      it wins with CIs** — no inert adapter, no fabricated contribution.
- [x] **Framework, multispectral, and communication claims are each backed by an executed run** or
      deleted. — no unsupported claims.
- [x] **Dataset choice satisfies the sponsor constraint** — emulate scale via partitioning one real
      dataset (Track A), with a real cross-domain option (Track B); no continental data collected.
- [x] **Reproducibility is enforced** (pinned env, seeds, saved partitions, hashes, one results
      store) — fixes the Colab-only / inconsistent-artifacts problems.
- [x] **Honesty section pre-empts reviewer objections** (simulated shift, easy dataset, taxonomy,
      privacy). — nothing oversold.

**Consistency spot-checks:** FedBN "keeps BN local / excludes from aggregation" (§6.2) is exactly
the fix for the BN buffer defect B6 (§5) — consistent. Track A+ *feature* shift is what justifies
FedBN (label skew alone would favor FedProx/personalization) — consistent, and E4/E5 ablate the two
shift types separately so the attribution is clean. Compute-budget parity (§7, G7) resolves the
regime-comparability defect (B9) and is enforced, not just described — consistent.

---

## 20. References (grounding)

- EuroSAT dataset & benchmark — [phelber/eurosat](https://github.com/phelber/eurosat),
  [HF EuroSAT_RGB](https://huggingface.co/datasets/blanchon/EuroSAT_RGB),
  [HF EuroSAT_MSI](https://huggingface.co/datasets/blanchon/EuroSAT_MSI),
  [PapersWithCode](https://paperswithcode.com/dataset/eurosat).
- Flower framework & simulation — [Quickstart PyTorch](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html),
  [Flower Datasets partitioners](https://flower.ai/docs/datasets/tutorial-use-partitioners.html),
  [DirichletPartitioner API](https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.DirichletPartitioner.html).
- Non-IID FL benchmark — Li et al., "Federated Learning on Non-IID Data Silos: An Experimental
  Study," ICDE 2022 — [paper](https://arxiv.org/pdf/2102.02079),
  [NIID-Bench code](https://github.com/Xtra-Computing/NIID-Bench).
- FedBN — Li et al., "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization,"
  ICLR 2021 — [arXiv](https://arxiv.org/abs/2102.07623),
  [Flower baseline](https://flower.ai/docs/baselines/fedbn.html).
- FedProx / SCAFFOLD / FedAdam — surveyed in the NIID-Bench study above and Flower strategy docs.
- Federated domain generalization / leave-one-client-out — FedFA
  ([arXiv:2301.12995](https://arxiv.org/pdf/2301.12995)), FedAlign
  ([arXiv:2501.15486](https://arxiv.org/html/2501.15486)).
- Cross-domain RS scene classification & shared taxonomy — Universal Domain Adaptation for RS
  ([arXiv:2301.11387](https://arxiv.org/pdf/2301.11387)); datasets
  [AID](https://captain-whu.github.io/AID/), [NWPU-RESISC45](https://gcheng-nwpu.github.io/),
  [UC-Merced](http://weegee.vision.ucmerced.edu/datasets/landuse.html).
- FL on space/edge platforms (motivation) — FEDGE, Int. J. Inf. Tech. 2025
  ([Springer](https://link.springer.com/article/10.1007/s41870-025-03010-0)).
