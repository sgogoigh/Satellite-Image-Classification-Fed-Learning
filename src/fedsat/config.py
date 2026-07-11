"""Typed experiment configuration with YAML round-trip.

One config object drives every regime so that comparisons stay controlled
(PLAN §2 "comparability", §7 budget parity). Persist it next to every result
(PLAN §11) so runs are reproducible and auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Tuple, List
import yaml


@dataclass
class ExperimentConfig:
    # --- identity / reproducibility ---
    experiment_name: str = "eurosat_fedsat"
    seed: int = 42

    # --- dataset ---
    dataset: str = "eurosat_rgb"          # 'eurosat_rgb' | 'eurosat_msi'
    hf_repo: str = "blanchon/EuroSAT_RGB"  # source of truth for labels (ClassLabel)
    num_classes: int = 10
    in_channels: int = 3                   # 13 for the multispectral track
    input_size: int = 64                   # 64 = fast (Colab-friendly); 224 = pretrained-native
    expected_total: int = 27000

    # --- partitioning / "regions" (PLAN §4) ---
    num_clients: int = 10
    alpha: float = 0.5                     # Dirichlet concentration: 100≈IID, 0.5 moderate, 0.1 severe
    global_test_frac: float = 0.15         # comparable held-out test, IID across all data
    val_frac: float = 0.15                 # per-client val (early stopping only)
    test_frac: float = 0.15                # per-client test (local-only + per-region analysis)
    min_client_size: int = 20              # Dirichlet redraw floor

    # --- model (PLAN §5) ---
    backbone: str = "resnet18"             # 'resnet18' | 'resnet50'
    pretrained: bool = True
    norm: str = "bn"                       # 'bn' | 'gn'  (FedBN handled in the FL strategy)

    # --- optimization (PLAN §6/§7: SGD, small local epochs) ---
    optimizer: str = "sgd"
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    batch_size: int = 64
    num_workers: int = 2
    lr_schedule: str = "cosine"            # 'cosine' | 'step' | 'none'

    # centralized / local-only budget
    max_epochs: int = 40
    early_stop_patience: int = 7
    early_stop_min_delta: float = 1e-4

    # federated budget (used from P2)
    num_rounds: int = 30
    local_epochs: int = 1
    fraction_fit: float = 1.0

    # --- paths (set to Drive in Colab for persistence) ---
    project_root: str = "."
    data_cache_dir: str = "data/hf_cache"
    partition_dir: str = "data/partitions"
    results_dir: str = "results"

    # --- device (filled at runtime) ---
    device: str = "cpu"

    # ------------------------------------------------------------------ #
    def partition_name(self) -> str:
        return f"{self.dataset}_K{self.num_clients}_alpha{self.alpha}_seed{self.seed}.json"

    def partition_path(self) -> Path:
        return Path(self.project_root) / self.partition_dir / self.partition_name()

    def run_dir(self) -> Path:
        return Path(self.project_root) / self.results_dir / self.experiment_name

    # ---- YAML round-trip ----
    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        known = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def replace(self, **kwargs) -> "ExperimentConfig":
        """Return a copy with fields overridden (for sweeps)."""
        d = asdict(self)
        d.update(kwargs)
        return ExperimentConfig(**d)
