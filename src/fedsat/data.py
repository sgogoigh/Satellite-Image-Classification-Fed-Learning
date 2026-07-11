"""Data layer: real EuroSAT loading, integrity gate, Dirichlet partitioning, splits.

Design goals (directly answering the audit, PLAN §3–§4):
  * REAL data only. There is deliberately **no synthetic fallback** anywhere in this
    module (kills B1/B2). If data is missing/short, we raise.
  * Labels come from the dataset's own ``ClassLabel`` names, never a hard-coded list
    (kills B3 — the "5 empty classes" bug).
  * Partitioning is deterministic and the resulting indices are SAVED to disk and
    reused by every regime (kills B7/B8). "Regions" are emulated by a Dirichlet
    label-skew split of one real dataset — no continent data is collected.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import numpy as np

from .config import ExperimentConfig
from .utils import data_fingerprint, disjoint, load_json, save_json, utc_now, git_commit

# ImageNet stats (standard for pretrained transfer; documented choice, PLAN §5)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# 1. Loading
# --------------------------------------------------------------------------- #
def load_eurosat(hf_repo: str = "blanchon/EuroSAT_RGB", cache_dir: Optional[str] = None):
    """Load the FULL EuroSAT dataset (all splits concatenated) from the HF Hub.

    Returns ``(hf_ds, class_names, labels)`` where ``hf_ds`` is a HF ``Dataset`` with
    'image' and 'label' columns, ``class_names`` is read from the label ``ClassLabel``
    feature, and ``labels`` is an int numpy array aligned with ``hf_ds`` row order.
    """
    from datasets import load_dataset, concatenate_datasets

    dd = load_dataset(hf_repo, cache_dir=cache_dir)
    # Concatenate whatever splits exist so we control the split ourselves (PLAN §4).
    parts = [dd[s] for s in dd.keys()]
    hf_ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]

    if "label" not in hf_ds.column_names:
        raise KeyError(f"'label' column not found in {hf_repo}; columns={hf_ds.column_names}")
    if "image" not in hf_ds.column_names:
        raise KeyError(f"'image' column not found in {hf_repo}; columns={hf_ds.column_names}")

    feat = hf_ds.features["label"]
    if hasattr(feat, "names") and feat.names:            # ClassLabel — the correct source
        class_names = list(feat.names)
    else:                                                # fallback: derive from data
        class_names = [str(v) for v in sorted(set(hf_ds["label"]))]

    labels = np.asarray(hf_ds["label"], dtype=np.int64)
    return hf_ds, class_names, labels


def integrity_gate(class_names, labels, expected_classes: int = 10,
                   expected_total: int = 27000, tol: int = 1000) -> dict:
    """Gate G1 (PLAN §12). Raise if the data is not what we require. No fallback exists.

    Returns a stats dict (per-class counts, total, data hash) for the EDA/provenance log.
    """
    labels = np.asarray(labels)
    total = int(labels.size)
    counts = {int(c): int((labels == c).sum()) for c in range(len(class_names))}

    assert len(class_names) == expected_classes, (
        f"G1 FAIL: expected {expected_classes} classes, found {len(class_names)}: {class_names}")
    empty = [class_names[c] for c, n in counts.items() if n == 0]
    assert not empty, f"G1 FAIL: these classes have ZERO samples: {empty} (the old B3 bug)"
    assert abs(total - expected_total) <= tol, (
        f"G1 FAIL: total={total} not within {tol} of expected {expected_total}")

    fp = data_fingerprint(class_names, counts, total)
    return {
        "total": total,
        "num_classes": len(class_names),
        "class_names": list(class_names),
        "class_counts": {class_names[c]: n for c, n in counts.items()},
        "data_hash": fp,
        "checked_at": utc_now(),
    }


# --------------------------------------------------------------------------- #
# 2. Splitting & partitioning (deterministic; PLAN §4)
# --------------------------------------------------------------------------- #
def stratified_split(indices, labels, val_frac: float, test_frac: float, seed: int):
    """Class-balanced train/val/test split of ``indices``. Robust to tiny per-class counts
    (unlike sklearn ``stratify`` which errors on singletons)."""
    rng = np.random.default_rng(seed)
    indices = np.asarray(indices)
    labs = np.asarray(labels)[indices]
    train, val, test = [], [], []
    for c in np.unique(labs):
        idx_c = indices[labs == c].copy()
        rng.shuffle(idx_c)
        n = len(idx_c)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        test.extend(idx_c[:n_test].tolist())
        val.extend(idx_c[n_test:n_test + n_val].tolist())
        train.extend(idx_c[n_test + n_val:].tolist())
    return sorted(train), sorted(val), sorted(test)


def dirichlet_partition(pool_indices, labels, num_clients: int, alpha: float,
                        seed: int, min_size: int = 20) -> dict[int, list[int]]:
    """Symmetric Dirichlet(alpha) label-distribution partition (NIID-Bench style).

    Each class's samples are divided among clients according to a per-class Dirichlet
    draw, producing controllable label skew. ``alpha`` large ⇒ ~IID; small ⇒ severe skew.
    Redraws until the smallest client has >= ``min_size`` samples.
    """
    labels = np.asarray(labels)
    pool = np.asarray(pool_indices)
    pool_labels = labels[pool]
    classes = np.unique(pool_labels)

    rng = np.random.default_rng(seed)
    for _attempt in range(100):
        buckets: list[list[int]] = [[] for _ in range(num_clients)]
        for c in classes:
            idx_c = pool[pool_labels == c].copy()
            rng.shuffle(idx_c)
            props = rng.dirichlet(np.full(num_clients, alpha))
            cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
            for i, part in enumerate(np.split(idx_c, cuts)):
                buckets[i].extend(part.tolist())
        if min(len(b) for b in buckets) >= min_size:
            break
    else:
        raise RuntimeError(
            f"Could not satisfy min_size={min_size} after 100 redraws; "
            f"lower alpha resolution or min_client_size.")
    return {i: sorted(b) for i, b in enumerate(buckets)}


def build_partition(cfg: ExperimentConfig, labels, class_names, data_hash: str) -> dict:
    """Produce the full, disjoint partition object and verify gate G2.

    Structure::

        { "meta": {...}, "global_test": [idx...],
          "clients": {"0": {"train":[...], "val":[...], "test":[...]}, ...} }
    """
    all_idx = np.arange(len(labels))

    # (a) comparable global test set (IID across all data), rest = pool
    pool, _empty, global_test = stratified_split(
        all_idx, labels, val_frac=0.0, test_frac=cfg.global_test_frac, seed=cfg.seed)

    # (b) Dirichlet-partition the pool across clients (the "regions")
    client_pool = dirichlet_partition(
        pool, labels, cfg.num_clients, cfg.alpha, seed=cfg.seed + 1,
        min_size=cfg.min_client_size)

    # (c) per-client stratified train/val/test
    clients: dict[str, dict] = {}
    for cid, idxs in client_pool.items():
        tr, va, te = stratified_split(
            idxs, labels, cfg.val_frac, cfg.test_frac, seed=cfg.seed + 100 + cid)
        clients[str(cid)] = {"train": tr, "val": va, "test": te}

    # (d) verify disjointness (gate G2)
    every = [global_test]
    for c in clients.values():
        every += [c["train"], c["val"], c["test"]]
    assert disjoint(*every), "G2 FAIL: split index sets are not pairwise disjoint"

    partition = {
        "meta": {
            "dataset": cfg.dataset,
            "hf_repo": cfg.hf_repo,
            "num_clients": cfg.num_clients,
            "alpha": cfg.alpha,
            "seed": cfg.seed,
            "global_test_frac": cfg.global_test_frac,
            "val_frac": cfg.val_frac,
            "test_frac": cfg.test_frac,
            "num_classes": cfg.num_classes,
            "class_names": list(class_names),
            "data_hash": data_hash,
            "git_commit": git_commit(cfg.project_root),
            "created_at": utc_now(),
            "config": asdict(cfg),
        },
        "global_test": list(map(int, global_test)),
        "clients": clients,
    }
    return partition


def save_partition(cfg: ExperimentConfig, partition: dict) -> str:
    path = cfg.partition_path()
    save_json(path, partition)
    return str(path)


def load_partition(cfg: ExperimentConfig) -> dict:
    return load_json(cfg.partition_path())


def partition_matrix(partition: dict, num_classes: int, labels) -> np.ndarray:
    """[num_classes x num_clients] count matrix over each client's TRAIN split (for the heatmap)."""
    labels = np.asarray(labels)
    clients = partition["clients"]
    mat = np.zeros((num_classes, len(clients)), dtype=int)
    for cid, splits in clients.items():
        j = int(cid)
        for i in splits["train"]:
            mat[int(labels[i]), j] += 1
    return mat


def label_entropy(counts: np.ndarray) -> float:
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


# --------------------------------------------------------------------------- #
# 3. Torch Dataset + transforms + loaders
# --------------------------------------------------------------------------- #
def build_transform(input_size: int, train: bool,
                    mean=IMAGENET_MEAN, std=IMAGENET_STD):
    from torchvision import transforms as T

    if train:
        return T.Compose([
            T.Resize((input_size, input_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


class SensorShift:
    """Deterministic per-client photometric/atmospheric transform (PIL -> PIL) that *emulates* a
    distinct sensor/atmosphere/season for a client — i.e. genuine **feature (covariate) shift** on
    top of the label-distribution shift (PLAN §3.2, Track A+). Factors are FIXED per client, so the
    shift is a stable domain, not augmentation noise. This is declared as simulated sensor variation.
    """

    def __init__(self, brightness=1.0, contrast=1.0, saturation=1.0, hue=0.0,
                 gamma=1.0, blur_sigma=0.0):
        self.brightness, self.contrast, self.saturation = brightness, contrast, saturation
        self.hue, self.gamma, self.blur_sigma = hue, gamma, blur_sigma

    def __call__(self, img):
        import torchvision.transforms.functional as F
        img = F.adjust_brightness(img, self.brightness)
        img = F.adjust_contrast(img, self.contrast)
        img = F.adjust_saturation(img, self.saturation)
        if abs(self.hue) > 1e-6:
            img = F.adjust_hue(img, self.hue)
        if abs(self.gamma - 1.0) > 1e-6:
            img = F.adjust_gamma(img, self.gamma)
        if self.blur_sigma > 0:
            img = F.gaussian_blur(img, kernel_size=3, sigma=self.blur_sigma)
        return img


def build_client_shifts(num_clients: int, seed: int = 0, strength: float = 1.0) -> dict:
    """Return ``{client_id_str: SensorShift}`` — a fixed, distinct feature shift per client.
    ``strength=0`` disables shift (returns empty dict → clients share the real distribution).
    """
    if strength <= 0:
        return {}
    rng = np.random.default_rng(seed)
    shifts = {}
    for cid in range(num_clients):
        shifts[str(cid)] = SensorShift(
            brightness=float(1.0 + strength * rng.uniform(-0.4, 0.4)),
            contrast=float(1.0 + strength * rng.uniform(-0.4, 0.4)),
            saturation=float(1.0 + strength * rng.uniform(-0.4, 0.4)),
            hue=float(strength * rng.uniform(-0.08, 0.08)),
            gamma=float(1.0 + strength * rng.uniform(-0.3, 0.3)),
            blur_sigma=float(max(0.0, strength * rng.uniform(-0.5, 1.2))),
        )
    return shifts


class IndexedHFDataset:
    """Torch-style dataset over a subset of a HF EuroSAT dataset.

    ``client_transform`` (optional) is applied to the PIL image BEFORE the tensor
    transform — this is the hook for the P4 per-client sensor-shift simulation. It is
    ``None`` here, so P0/P1 use unaltered real imagery.
    """

    def __init__(self, hf_ds, indices, transform, client_transform=None):
        self.hf_ds = hf_ds
        self.indices = list(map(int, indices))
        self.transform = transform
        self.client_transform = client_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        rec = self.hf_ds[self.indices[i]]
        img = rec["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.client_transform is not None:
            img = self.client_transform(img)
        x = self.transform(img)
        y = int(rec["label"])
        return x, y


def make_loader(hf_ds, indices, cfg: ExperimentConfig, train: bool,
                shuffle: Optional[bool] = None, client_transform=None):
    from torch.utils.data import DataLoader

    tf = build_transform(cfg.input_size, train=train)
    ds = IndexedHFDataset(hf_ds, indices, tf, client_transform=client_transform)
    if shuffle is None:
        shuffle = train
    # persistent_workers keeps workers alive across the per-epoch iterator re-creation,
    # which avoids the harmless-but-noisy "_MultiProcessingDataLoaderIter.__del__ ...
    # can only test a child process" teardown spam seen in Colab notebooks.
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers, pin_memory=(cfg.device == "cuda"),
                      persistent_workers=(cfg.num_workers > 0), drop_last=False)


def pooled_indices(partition: dict, split: str) -> list[int]:
    """Union of a given split ('train'|'val'|'test') across all clients (for centralized)."""
    out: list[int] = []
    for c in partition["clients"].values():
        out += c[split]
    return sorted(out)
