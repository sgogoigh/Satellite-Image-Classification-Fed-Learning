"""Federated averaging — transparent, correct reference implementation (McMahan et al. 2017).

This is the validated FL core used to clear **Gate G4** (FedAvg ≈ centralized on IID). It fixes
the exact defects the audit found in the old pipeline:
  * clients optimize with **SGD** for a small number of **local epochs** (not Adam × many) — B5;
  * the **global aggregated model** is evaluated every round (not locally-refit clients) — B4/G5;
  * round selection uses a **held-out validation** signal, and TEST is reported at the selected
    round — no test-set peeking (B7 spirit); the returned model IS the selected model (B10/B19);
  * **gradient-step / epoch-equivalent budget** is tracked so regimes can be compared iso-compute
    (B9/G7); real **communication bytes** per round are logged (B14);
  * all shared tensors are aggregated (BatchNorm buffers included) with a documented policy — B6
    (FedBN, i.e. keeping BN local, is added as an option in P4).

The P2 notebook additionally runs the *same* FedAvg through the Flower framework and checks parity,
so the thesis can cite Flower without the make-or-break gate depending on framework version churn.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .engine import build_optimizer, evaluate
from .models import build_model
from .data import make_loader, pooled_indices


def _clone_state(model) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def state_nbytes(state: dict) -> int:
    """Total bytes of a model state (for honest communication-cost accounting)."""
    return int(sum(t.numel() * t.element_size() for t in state.values()))


def bn_state_keys(model) -> set:
    """State-dict keys belonging to BatchNorm layers (weight/bias/running_mean/running_var/
    num_batches_tracked). FedBN keeps exactly these **local** (never aggregated)."""
    keys = set()
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            for pname, _ in module.named_parameters(recurse=False):
                keys.add(f"{name}.{pname}")
            for bname, _ in module.named_buffers(recurse=False):
                keys.add(f"{name}.{bname}")
    return keys


def _local_train(model, loader, cfg, device, local_epochs: int, mu: float = 0.0):
    """One client's local update. If ``mu > 0`` adds the FedProx proximal term
    ``(mu/2) * ||w - w_global||^2`` over the trainable weights (McMahan-style FedAvg when mu==0,
    Li et al. 2020 FedProx when mu>0). Returns the number of gradient steps taken.
    """
    optimizer = build_optimizer(model, cfg)
    criterion = nn.CrossEntropyLoss()
    global_ref = [p.detach().clone() for p in model.parameters()] if mu > 0 else None
    steps = 0
    for _ in range(local_epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            if mu > 0:
                prox = sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), global_ref))
                loss = loss + (mu / 2.0) * prox
            loss.backward()
            optimizer.step()
            steps += 1
    return steps


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


def _compress_delta(delta, method: str, ratio: float):
    """Compress one tensor's client->server delta and return ``(applied_delta, uplink_bytes)``.

      * ``topk``  : keep the top ``ratio`` fraction of entries by magnitude (COO: 4B value + 4B idx).
      * ``8bit``  : per-tensor symmetric int8 quantization (1B/elem + 4B scale).
      * else      : no compression (4B/elem float32).

    The compression genuinely perturbs the update, so its accuracy cost is real (not just accounting).
    """
    numel = delta.numel()
    if method == "topk":
        k = max(1, int(round(ratio * numel)))
        flat = delta.flatten()
        idx = torch.topk(flat.abs(), k).indices
        comp = torch.zeros_like(flat)
        comp[idx] = flat[idx]
        return comp.view_as(delta), k * 8
    if method in ("8bit", "qsgd8"):
        maxv = float(delta.abs().max())
        if maxv == 0.0:
            return delta.clone(), numel + 4
        scale = maxv / 127.0
        q = torch.round(delta / scale).clamp_(-127, 127)
        return q * scale, numel + 4
    return delta.clone(), numel * 4


def _compressed_aggregate(states, sizes, pre_global, method, ratio):
    """FedAvg on **compressed deltas** (uplink). Returns ``(aggregated_state, uplink_bytes)``.
    Float tensors are delta-compressed; integer buffers (e.g. num_batches_tracked) are copied."""
    total = float(sum(sizes))
    agg, up_bytes = {}, 0
    for k, t0 in states[0].items():
        if t0.is_floating_point():
            acc = torch.zeros_like(t0, dtype=torch.float64)
            for st, n in zip(states, sizes):
                comp, nb = _compress_delta(st[k] - pre_global[k], method, ratio)
                acc += comp.double() * (n / total)
                up_bytes += nb
            agg[k] = (pre_global[k].double() + acc).to(t0.dtype)
        else:
            j = int(max(range(len(sizes)), key=lambda i: sizes[i]))
            agg[k] = states[j][k].clone()
            up_bytes += states[j][k].numel() * states[j][k].element_size()
    return agg, up_bytes


def run_fedavg(cfg, hf_ds, partition, class_names,
               num_rounds: int, local_epochs: int, fraction_fit: float = 1.0,
               mu: float = 0.0, compress: str = None, compress_ratio: float = 0.1,
               verbose: bool = True):
    """Run FedAvg (``mu==0``) or **FedProx** (``mu>0``, proximal term) over the saved partition.

    Selection is by **global validation accuracy** (union of client val splits) — the reported
    TEST metrics come from the round with the best val accuracy, and the returned model is that
    same model (no test peeking; saved == reported). Compute is tracked in gradient steps and
    epoch-equivalents for iso-budget comparison with the centralized baseline.

    ``compress`` (``None`` | ``'topk'`` | ``'8bit'``) applies real uplink compression at
    ``compress_ratio`` (top-k fraction) and logs the reduced communication (P5/E8). ``compress=None``
    is byte-for-byte the original behaviour used in P2/P3.

    Returns ``(global_model_at_best, history, summary)``.
    """
    device = cfg.device

    def new_model():
        return build_model(cfg.backbone, cfg.num_classes, cfg.pretrained,
                           cfg.in_channels, cfg.norm).to(device)

    global_model = new_model()
    local_model = new_model()                      # reused scratch model (reset each use)
    global_state = _clone_state(global_model)      # CPU master copy of the shared weights

    client_ids = sorted(partition["clients"].keys(), key=int)
    train_loaders = {cid: make_loader(hf_ds, partition["clients"][cid]["train"], cfg, train=True)
                     for cid in client_ids}
    train_sizes = {cid: len(partition["clients"][cid]["train"]) for cid in client_ids}
    steps_per_client = {cid: len(train_loaders[cid]) for cid in client_ids}   # batches == grad steps / local epoch

    val_loader = make_loader(hf_ds, pooled_indices(partition, "val"), cfg, train=False)   # SELECTION (no peeking)
    gtest_loader = make_loader(hf_ds, partition["global_test"], cfg, train=False)         # REPORTING

    total_train = sum(train_sizes.values())
    steps_per_epoch_equiv = max(1, int(np.ceil(total_train / cfg.batch_size)))
    model_bytes = state_nbytes(global_state)
    model_mb = model_bytes / (1024 ** 2)

    rng = np.random.default_rng(cfg.seed + 999)
    history: list[dict] = []
    comm_cum, grad_steps_cum = 0.0, 0
    best = {"val_acc": -1.0, "round": 0, "state": None, "test_metrics": None}

    for rnd in range(num_rounds):
        # ---- client selection (partial participation supported) ----
        if fraction_fit >= 1.0:
            selected = client_ids
        else:
            m = max(1, int(round(fraction_fit * len(client_ids))))
            selected = sorted(rng.choice(client_ids, size=m, replace=False).tolist(), key=int)

        # ---- broadcast global -> local SGD (FedAvg) or proximal SGD (FedProx) per client ----
        pre_global = {k: v.clone() for k, v in global_state.items()} if compress else None
        states, sizes = [], []
        for cid in selected:
            local_model.load_state_dict({k: v.to(device) for k, v in global_state.items()})
            _local_train(local_model, train_loaders[cid], cfg, device, local_epochs, mu=mu)
            states.append(_clone_state(local_model))
            sizes.append(train_sizes[cid])
            grad_steps_cum += steps_per_client[cid] * local_epochs

        # ---- aggregate (FedAvg); compress the uplink if requested ----
        if compress:
            global_state, up_bytes = _compressed_aggregate(states, sizes, pre_global, compress, compress_ratio)
        else:
            global_state = fedavg_aggregate(states, sizes)
            up_bytes = len(selected) * model_bytes
        comm_round = (up_bytes + len(selected) * model_bytes) / (1024 ** 2)   # compressed uplink + full downlink
        comm_cum += comm_round

        # ---- evaluate the GLOBAL model: val (for selection) + test (for reporting/curve) ----
        global_model.load_state_dict({k: v.to(device) for k, v in global_state.items()})
        val_m = evaluate(global_model, val_loader, device, cfg.num_classes, class_names)
        test_m = evaluate(global_model, gtest_loader, device, cfg.num_classes, class_names)
        history.append({
            "round": rnd + 1,
            "val_accuracy": val_m["accuracy"], "val_macro_f1": val_m["macro_f1"],
            "test_accuracy": test_m["accuracy"], "test_macro_f1": test_m["macro_f1"],
            "test_cohen_kappa": test_m["cohen_kappa"], "n_selected": len(selected),
            "grad_steps_cumulative": grad_steps_cum,
            "epoch_equiv_cumulative": grad_steps_cum / steps_per_epoch_equiv,
            "comm_mb_round": comm_round, "comm_mb_cumulative": comm_cum,
        })
        if verbose:
            print(f"  round {rnd+1:>2}/{num_rounds}  val_acc={val_m['accuracy']:.4f}  "
                  f"test_acc={test_m['accuracy']:.4f}  (~{grad_steps_cum/steps_per_epoch_equiv:.1f} "
                  f"epoch-equiv, {comm_cum:.0f}MB)", flush=True)

        # ---- track best round by VALIDATION accuracy ----
        if val_m["accuracy"] > best["val_acc"]:
            best = {"val_acc": val_m["accuracy"], "round": rnd + 1,
                    "state": {k: v.clone() for k, v in global_state.items()}, "test_metrics": test_m}

    # restore the best-by-val model (this is what gets returned + saved)
    if best["state"] is not None:
        global_model.load_state_dict({k: v.to(device) for k, v in best["state"].items()})

    summary = {
        "select_by": "val",
        "algorithm": "fedprox" if mu > 0 else "fedavg",
        "mu": mu, "compress": compress, "compress_ratio": compress_ratio,
        "num_clients": len(client_ids), "fraction_fit": fraction_fit,
        "best_round": best["round"],
        "best_val_accuracy": best["val_acc"],
        "test_accuracy_at_best": best["test_metrics"]["accuracy"],
        "best_metrics": best["test_metrics"],
        "final_round_test_accuracy": history[-1]["test_accuracy"],
        "total_grad_steps": grad_steps_cum,
        "epoch_equivalents": grad_steps_cum / steps_per_epoch_equiv,
        "total_comm_mb": comm_cum, "model_mb": model_mb,
        "rounds": num_rounds, "local_epochs": local_epochs, "fraction_fit": fraction_fit,
    }
    return global_model, history, summary


def run_federated(cfg, hf_ds, partition, class_names, num_rounds, local_epochs,
                  fraction_fit: float = 1.0, mu: float = 0.0, keep_bn_local: bool = False,
                  client_transforms=None, verbose: bool = True):
    """Unified federated runner scored by **per-client own-test accuracy** (the deployment-relevant
    metric under feature shift, and the only metric well-defined for FedBN).

      * ``keep_bn_local=False`` → FedAvg (``mu==0``) or FedProx (``mu>0``): a single global model.
      * ``keep_bn_local=True``  → **FedBN**: BatchNorm layers are kept per-client (never aggregated);
        only non-BN weights are averaged. Each client is evaluated with the shared weights + its own
        BN (a personalized model).

    ``client_transforms``: optional ``{client_id_str: callable}`` applied to each client's PIL images
    (the per-client sensor-shift simulation) so clients differ in *feature* distribution.

    Selection is by **mean per-client validation accuracy** (no test peeking). Returns
    ``(history, summary)`` with mean/worst per-client test accuracy at the selected round.
    """
    device = cfg.device

    def new_model():
        return build_model(cfg.backbone, cfg.num_classes, cfg.pretrained,
                           cfg.in_channels, cfg.norm).to(device)

    scratch = new_model()
    bn_keys = bn_state_keys(scratch) if keep_bn_local else set()
    ctf = client_transforms or {}

    client_ids = sorted(partition["clients"].keys(), key=int)
    train_loaders = {c: make_loader(hf_ds, partition["clients"][c]["train"], cfg, train=True,
                                    client_transform=ctf.get(c)) for c in client_ids}
    val_loaders = {c: make_loader(hf_ds, partition["clients"][c]["val"], cfg, train=False,
                                  client_transform=ctf.get(c)) for c in client_ids}
    test_loaders = {c: make_loader(hf_ds, partition["clients"][c]["test"], cfg, train=False,
                                   client_transform=ctf.get(c)) for c in client_ids}
    train_sizes = {c: len(partition["clients"][c]["train"]) for c in client_ids}

    global_state = _clone_state(scratch)
    client_state = {c: {k: v.clone() for k, v in global_state.items()} for c in client_ids}

    def personalized_state(c):
        if keep_bn_local:
            return {k: (client_state[c][k] if k in bn_keys else global_state[k]) for k in global_state}
        return global_state

    def eval_client(c, loader):
        scratch.load_state_dict({k: v.to(device) for k, v in personalized_state(c).items()})
        return evaluate(scratch, loader, device, cfg.num_classes, class_names)

    model_mb = state_nbytes(global_state) / (1024 ** 2)
    steps_per_epoch_equiv = max(1, int(np.ceil(sum(train_sizes.values()) / cfg.batch_size)))
    rng = np.random.default_rng(cfg.seed + 999)
    history, comm_cum, grad_steps = [], 0.0, 0
    best = {"val": -1.0, "round": 0, "global": None, "client": None}

    for rnd in range(num_rounds):
        if fraction_fit >= 1.0:
            selected = client_ids
        else:
            m = max(1, int(round(fraction_fit * len(client_ids))))
            selected = sorted(rng.choice(client_ids, size=m, replace=False).tolist(), key=int)

        states, sizes = [], []
        for c in selected:
            scratch.load_state_dict({k: v.to(device) for k, v in personalized_state(c).items()})
            _local_train(scratch, train_loaders[c], cfg, device, local_epochs, mu=mu)
            client_state[c] = _clone_state(scratch)   # persist full state (incl this client's BN)
            states.append(client_state[c]); sizes.append(train_sizes[c])
            grad_steps += len(train_loaders[c]) * local_epochs

        agg = fedavg_aggregate(states, sizes)
        if keep_bn_local:
            for k in global_state:                     # FedBN: aggregate only non-BN keys
                if k not in bn_keys:
                    global_state[k] = agg[k]
        else:
            global_state = agg
        comm_cum += 2.0 * len(selected) * model_mb

        val_accs = [eval_client(c, val_loaders[c])["accuracy"] for c in client_ids]
        mean_val = float(np.mean(val_accs))
        history.append({"round": rnd + 1, "mean_val_acc": mean_val,
                        "comm_mb_cumulative": comm_cum,
                        "epoch_equiv_cumulative": grad_steps / steps_per_epoch_equiv})
        if verbose:
            print(f"  round {rnd+1:>2}/{num_rounds}  mean_val_acc={mean_val:.4f}  "
                  f"(~{grad_steps/steps_per_epoch_equiv:.1f} epoch-equiv, {comm_cum:.0f}MB)", flush=True)

        if mean_val > best["val"]:
            best = {"val": mean_val, "round": rnd + 1,
                    "global": {k: v.clone() for k, v in global_state.items()},
                    "client": ({c: {k: v.clone() for k, v in client_state[c].items()} for c in client_ids}
                               if keep_bn_local else None)}

    # restore best-by-val states, then evaluate per-client TEST once
    if best["global"] is not None:
        global_state = best["global"]
        if keep_bn_local and best["client"] is not None:
            client_state = best["client"]
    per_client = []
    for c in client_ids:
        acc = eval_client(c, test_loaders[c])["accuracy"]
        per_client.append({"client": c, "test_acc": acc, "n_train": train_sizes[c]})
    test_accs = [p["test_acc"] for p in per_client]

    summary = {
        "algorithm": "fedbn" if keep_bn_local else ("fedprox" if mu > 0 else "fedavg"),
        "keep_bn_local": keep_bn_local, "mu": mu, "norm": cfg.norm,
        "best_round": best["round"], "best_mean_val_acc": best["val"],
        "mean_test_acc": float(np.mean(test_accs)), "worst_test_acc": float(np.min(test_accs)),
        "std_test_acc": float(np.std(test_accs)), "per_client_test": per_client,
        "total_comm_mb": comm_cum, "epoch_equivalents": grad_steps / steps_per_epoch_equiv,
        "rounds": num_rounds, "local_epochs": local_epochs,
    }
    return history, summary
