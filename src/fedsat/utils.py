"""Small cross-cutting helpers: seeding, hashing, device, IO, provenance."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python / numpy / torch (+ cuda). Enables determinism gate G6."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def sha256_of(obj: Any) -> str:
    """Stable SHA-256 of a JSON-serializable object (for data/partition provenance)."""
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def data_fingerprint(class_names: list[str], label_counts: dict[int, int], total: int) -> str:
    """Short hash summarizing the dataset actually loaded (records B19 provenance)."""
    return sha256_of(
        {"classes": list(class_names), "counts": {str(k): int(v) for k, v in label_counts.items()},
         "total": int(total)}
    )[:16]


def git_commit(root: str | Path = ".") -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def disjoint(*index_lists: Iterable[int]) -> bool:
    """True iff all provided index collections are pairwise disjoint (gate G2)."""
    seen: set[int] = set()
    for lst in index_lists:
        s = set(int(i) for i in lst)
        if s & seen:
            return False
        seen |= s
    return True
