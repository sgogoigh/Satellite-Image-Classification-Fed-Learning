"""Training & evaluation engine for centralized / local-only regimes.

All diagnostics (accuracy, macro-F1, per-class F1, confusion, kappa) are computed by a
single ``evaluate`` on the TEST set from the SAME model under test — never a throwaway
probe on val (kills B7/B10). Early stopping uses val only. Clients use SGD + schedule
(PLAN §6/§7); the same routines back the FL client update in later phases.
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def build_optimizer(model, cfg):
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                               weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"unknown optimizer: {cfg.optimizer}")


def build_scheduler(optimizer, cfg, total_epochs: int):
    if cfg.lr_schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs))
    if cfg.lr_schedule == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, total_epochs // 3), gamma=0.1)
    return None


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int, class_names=None) -> dict:
    """Full metric suite on whatever split `loader` holds (use the TEST loader for reports)."""
    from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                                  cohen_kappa_score, balanced_accuracy_score)

    model.eval()
    preds, labs = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds.append(logits.argmax(1).cpu().numpy())
        labs.append(y.numpy())
    if not preds:
        return {"accuracy": float("nan"), "n": 0}
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(labs)
    labels_range = list(range(num_classes))
    per_class = f1_score(y_true, y_pred, average=None, labels=labels_range, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels_range)
    out = {
        "n": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels_range, zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=labels_range)),
        "per_class_f1": {(class_names[c] if class_names else str(c)): float(per_class[c])
                         for c in range(num_classes)},
        "confusion_matrix": cm.tolist(),
    }
    return out


def fit(model, train_loader, val_loader, cfg, device, criterion=None, verbose=True) -> dict:
    """Train with early stopping on val accuracy; restore best weights. Returns history.

    Reports gradient-step / epoch counts so budget parity (gate G7) is auditable.
    """
    criterion = criterion or nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, cfg.max_epochs)

    best_acc, best_state, stale = -1.0, None, 0
    history = {"train_loss": [], "val_acc": [], "epochs_run": 0, "grad_steps": 0}
    steps_per_epoch = len(train_loader)

    for epoch in range(cfg.max_epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        if scheduler is not None:
            scheduler.step()
        val = evaluate(model, val_loader, device, cfg.num_classes)
        acc = val["accuracy"]
        history["train_loss"].append(loss)
        history["val_acc"].append(acc)
        history["epochs_run"] = epoch + 1
        history["grad_steps"] += steps_per_epoch
        if verbose:
            print(f"  epoch {epoch+1:>2}/{cfg.max_epochs}  train_loss={loss:.4f}  val_acc={acc:.4f}")
        if acc > best_acc + cfg.early_stop_min_delta:
            best_acc, stale = acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.early_stop_patience:
                if verbose:
                    print(f"  early stop at epoch {epoch+1} (best val_acc={best_acc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    history["best_val_acc"] = best_acc
    return history


@torch.no_grad()
def _acc(model, x, y):
    return float((model(x).argmax(1) == y).float().mean().item())


def overfit_one_batch(model, loader, device, steps: int = 60, lr: float = 0.01) -> float:
    """Gate G3: a correct model+loss should reach ~1.0 accuracy on a single batch."""
    model.train()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
    model.eval()
    return _acc(model, x, y)
