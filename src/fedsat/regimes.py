"""Non-federated baselines: centralized (upper bound) and local-only (lower bound).

Both are evaluated on the SAME comparable **global test set** used by the federated regimes, with
selection by validation (no test peeking) — so every regime in the P3 sweep is measured on equal
footing (PLAN §6, §8). Local-only additionally reports each client's own in-domain test accuracy,
which contrasts sharply with its global-test accuracy under heterogeneity.
"""
from __future__ import annotations

import numpy as np

from .data import make_loader, pooled_indices
from .engine import evaluate, fit
from .models import build_model


def run_centralized(cfg, hf_ds, partition, class_names):
    """Train one model on the UNION of client train splits (α-independent: the pool is the same
    regardless of how it is partitioned). Report global-test metrics. This is the upper bound."""
    device = cfg.device
    train_idx = pooled_indices(partition, "train")
    val_idx = pooled_indices(partition, "val")
    train_loader = make_loader(hf_ds, train_idx, cfg, train=True)
    val_loader = make_loader(hf_ds, val_idx, cfg, train=False)
    gtest_loader = make_loader(hf_ds, partition["global_test"], cfg, train=False)

    model = build_model(cfg.backbone, cfg.num_classes, cfg.pretrained, cfg.in_channels, cfg.norm).to(device)
    hist = fit(model, train_loader, val_loader, cfg, device, progress=False, verbose=False)
    m = evaluate(model, gtest_loader, device, cfg.num_classes, class_names)
    summary = {
        "regime": "centralized",
        "global_test_accuracy": m["accuracy"],
        "global_test_macro_f1": m["macro_f1"],
        "best_val_acc": hist["best_val_acc"],
        "grad_steps": hist["grad_steps"],
        "epochs_run": hist["epochs_run"],
        "metrics": m,
    }
    return model, summary


def run_local_only(cfg, hf_ds, partition, class_names, local_max_epochs=None, verbose=False):
    """Each client trains ONLY on its own data (same pretrained init), early-stops on its own val.
    Each local model is evaluated on (a) the global test set — how well a lone client generalizes to
    the full distribution (the lower bound that collapses under heterogeneity), and (b) its own test
    set — in-domain performance (which stays high even under skew). Returns mean/worst across clients.
    """
    device = cfg.device
    cfg_local = cfg.replace(max_epochs=local_max_epochs) if local_max_epochs else cfg
    gtest_loader = make_loader(hf_ds, partition["global_test"], cfg, train=False)

    per_client = []
    for cid, splits in sorted(partition["clients"].items(), key=lambda kv: int(kv[0])):
        model = build_model(cfg.backbone, cfg.num_classes, cfg.pretrained, cfg.in_channels, cfg.norm).to(device)
        tl = make_loader(hf_ds, splits["train"], cfg, train=True)
        vl = make_loader(hf_ds, splits["val"], cfg, train=False)
        fit(model, tl, vl, cfg_local, device, progress=False, verbose=False)
        g = evaluate(model, gtest_loader, device, cfg.num_classes, class_names)["accuracy"]
        own = evaluate(model, make_loader(hf_ds, splits["test"], cfg, train=False),
                       device, cfg.num_classes, class_names)["accuracy"]
        per_client.append({"client": cid, "global_test_acc": g, "own_test_acc": own,
                           "n_train": len(splits["train"])})
        if verbose:
            print(f"  client {cid}: global_test={g:.4f}  own_test={own:.4f}", flush=True)

    g_accs = [p["global_test_acc"] for p in per_client]
    o_accs = [p["own_test_acc"] for p in per_client]
    summary = {
        "regime": "local_only",
        "mean_global_test_accuracy": float(np.mean(g_accs)),
        "worst_global_test_accuracy": float(np.min(g_accs)),
        "mean_own_test_accuracy": float(np.mean(o_accs)),
        "worst_own_test_accuracy": float(np.min(o_accs)),
        "per_client": per_client,
    }
    return summary
