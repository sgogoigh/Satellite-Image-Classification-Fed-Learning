"""Federated averaging — transparent, correct reference implementation (McMahan et al. 2017).

This is the validated FL core used to clear **Gate G4** (FedAvg ≈ centralized on IID). It fixes
the exact defects the audit found in the old pipeline:
  * clients optimize with **SGD** for a small number of **local epochs** (not Adam × many) — B5;
  * the **global aggregated model** is evaluated every round on the held-out global test set
    (not locally-refit client models) — B4/B5/G5;
  * **all** shared tensors are aggregated, including BatchNorm running buffers, with a documented
    policy — B6 (FedBN, i.e. keeping BN local, is added as an option in P4);
  * real **communication bytes** per round are logged — B14.

The P2 notebook additionally runs the *same* FedAvg through the Flower framework and checks parity,
so the thesis can cite Flower without the make-or-break gate depending on framework version churn.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .engine import build_optimizer, evaluate, train_one_epoch
from .models import build_model
from .data import make_loader


def _clone_state(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def state_nbytes(state: dict) -> int:
    """Total bytes of a model state (for honest communication-cost accounting)."""
    return int(sum(t.numel() * t.element_size() for t in state.values()))


def fedavg_aggregate(states: list[dict], sizes: list[int]) -> dict:
    """Sample-weighted average of client states (FedAvg).

    Float tensors (weights + BN running_mean/var) are averaged in float64 then cast back.
    Non-float buffers (e.g. BatchNorm ``num_batches_tracked``, int64) are taken from the
    largest client rather than averaged, to keep dtypes valid.
    """
    total = float(sum(sizes))
    agg: dict = {}
    for k, t0 in states[0].items():
        if t0.is_floating_point():
            acc = torch.zeros_like(t0, dtype=torch.float64)
            for st, n in zip(states, sizes):
                acc += st[k].double() * (n / total)
            agg[k] = acc.to(t0.dtype)
        else:
            j = int(max(range(len(sizes)), key=lambda i: sizes[i]))
            agg[k] = states[j][k].clone()
    return agg


def run_fedavg(cfg, hf_ds, partition, class_names,
               num_rounds: int, local_epochs: int, fraction_fit: float = 1.0,
               verbose: bool = True):
    """Run FedAvg over the saved partition; evaluate the GLOBAL model each round on the
    global test set. Returns ``(global_model, history, summary)``.

    ``history``: per-round list of {round, accuracy, macro_f1, cohen_kappa, comm_mb_round,
    comm_mb_cumulative, n_selected}. ``summary``: best round + final metrics.
    """
    device = cfg.device
    criterion = nn.CrossEntropyLoss()

    def new_model():
        return build_model(cfg.backbone, cfg.num_classes, cfg.pretrained,
                           cfg.in_channels, cfg.norm).to(device)

    global_model = new_model()
    local_model = new_model()                      # reused scratch model (fast; reset each use)
    global_state = _clone_state(global_model)      # CPU master copy of the shared weights

    client_ids = sorted(partition["clients"].keys(), key=int)
    train_loaders = {cid: make_loader(hf_ds, partition["clients"][cid]["train"], cfg, train=True)
                     for cid in client_ids}
    train_sizes = {cid: len(partition["clients"][cid]["train"]) for cid in client_ids}
    gtest_loader = make_loader(hf_ds, partition["global_test"], cfg, train=False)

    model_mb = state_nbytes(global_state) / (1024 ** 2)
    rng = np.random.default_rng(cfg.seed + 999)
    history: list[dict] = []
    comm_cum = 0.0

    for rnd in range(num_rounds):
        # ---- client selection (partial participation supported) ----
        if fraction_fit >= 1.0:
            selected = client_ids
        else:
            m = max(1, int(round(fraction_fit * len(client_ids))))
            selected = sorted(rng.choice(client_ids, size=m, replace=False).tolist(), key=int)

        # ---- broadcast global -> local training on each selected client ----
        states, sizes = [], []
        for cid in selected:
            local_model.load_state_dict({k: v.to(device) for k, v in global_state.items()})
            optimizer = build_optimizer(local_model, cfg)          # fresh optimizer each round (no stale momentum)
            for _ in range(local_epochs):
                train_one_epoch(local_model, train_loaders[cid], optimizer, criterion, device)
            states.append(_clone_state(local_model))
            sizes.append(train_sizes[cid])

        # ---- aggregate (FedAvg) ----
        global_state = fedavg_aggregate(states, sizes)

        # ---- communication for the round: download + upload of the shared model ----
        comm_round = 2.0 * len(selected) * model_mb
        comm_cum += comm_round

        # ---- evaluate the GLOBAL model on the held-out global test set (G5) ----
        global_model.load_state_dict({k: v.to(device) for k, v in global_state.items()})
        m = evaluate(global_model, gtest_loader, device, cfg.num_classes, class_names)
        row = {"round": rnd + 1, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
               "cohen_kappa": m["cohen_kappa"], "n_selected": len(selected),
               "comm_mb_round": comm_round, "comm_mb_cumulative": comm_cum}
        history.append(row)
        if verbose:
            print(f"  round {rnd+1:>2}/{num_rounds}  global_test_acc={m['accuracy']:.4f}  "
                  f"macro_f1={m['macro_f1']:.4f}  comm={comm_cum:.0f}MB", flush=True)

    best = max(history, key=lambda r: r["accuracy"])
    final_metrics = evaluate(global_model, gtest_loader, device, cfg.num_classes, class_names)
    summary = {"best_round": best["round"], "best_accuracy": best["accuracy"],
               "final_accuracy": final_metrics["accuracy"], "final_metrics": final_metrics,
               "total_comm_mb": comm_cum, "model_mb": model_mb, "rounds": num_rounds,
               "local_epochs": local_epochs, "fraction_fit": fraction_fit}
    return global_model, history, summary
