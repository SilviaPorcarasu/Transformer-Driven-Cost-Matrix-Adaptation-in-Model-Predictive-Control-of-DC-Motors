from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import MPCDataset, get_manifest
from model import HybridControllerModel


def setup_logging(out_dir: str) -> logging.Logger:
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(os.path.join(out_dir, "train.log"), mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def frob_rel(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    num = torch.linalg.norm(a - b, ord="fro", dim=(-2, -1))
    den = torch.linalg.norm(b, ord="fro", dim=(-2, -1)) + eps
    return (num / den).mean()


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ((a - b) ** 2).mean()


def regression_loss(a: torch.Tensor, b: torch.Tensor, weights: Dict[str, float], component: str = "") -> torch.Tensor:
    loss_name = weights.get(f"target_loss_{component}") or weights.get("target_loss")
    if loss_name == "huber":
        return nn.functional.smooth_l1_loss(a, b, beta=float(weights.get("huber_beta", 0.25)))
    return mse(a, b)


def safe_log10_diag(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.log10(torch.diagonal(x, dim1=-2, dim2=-1).clamp_min(eps))


def rel_diag_from_log(pred_log: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    pred = torch.pow(10.0, pred_log)
    true = torch.diagonal(target, dim1=-2, dim2=-1)
    return (torch.linalg.norm(pred - true, dim=-1) / (torch.linalg.norm(true, dim=-1) + eps)).mean()


def diag_matrix_from_log(pred_log: torch.Tensor) -> torch.Tensor:
    return torch.diag_embed(torch.pow(10.0, pred_log))


def riccati_residual(P: torch.Tensor, Q: torch.Tensor, R: torch.Tensor,
                     A: torch.Tensor, B: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    eye = torch.eye(R.shape[-1], device=R.device, dtype=R.dtype).expand_as(R)
    AtP = A.transpose(-1, -2) @ P
    BtP = B.transpose(-1, -2) @ P
    M = R + BtP @ B + eps * eye
    gain_term = AtP @ B @ torch.linalg.solve(M, BtP @ A)
    residual = P - Q - AtP @ A + gain_term
    return torch.linalg.norm(residual, ord="fro", dim=(-2, -1)).mean()


def dare_iterate(A: torch.Tensor, B: torch.Tensor, Q: torch.Tensor, R: torch.Tensor,
                 n_iter: int = 80, eps: float = 1e-6) -> torch.Tensor:
    """Approximate the stabilizing DARE solution with differentiable Riccati iterations."""
    P = Q
    eye = torch.eye(R.shape[-1], device=R.device, dtype=R.dtype).expand_as(R)
    for _ in range(n_iter):
        AtP = A.transpose(-1, -2) @ P
        BtP = B.transpose(-1, -2) @ P
        M = R + BtP @ B + eps * eye
        P_next = Q + AtP @ A - AtP @ B @ torch.linalg.solve(M, BtP @ A)
        P = 0.5 * (P_next + P_next.transpose(-1, -2))
    return P


def compute_normalization(ds: MPCDataset, indices: Iterable[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seq_values = []
    static_values = []
    for idx in indices:
        item = ds.raw_tokens_for_stats(int(idx))
        seq_values.append(item["tokens_seq"])
        static_values.append(item["tokens_static"])

    seq = np.concatenate(seq_values, axis=0)
    static = np.stack(static_values, axis=0)
    seq_mean = seq.mean(axis=0).astype(np.float32)
    seq_std = np.maximum(seq.std(axis=0).astype(np.float32), 1e-6)
    static_mean = static.mean(axis=0).astype(np.float32)
    static_std = np.maximum(static.std(axis=0).astype(np.float32), 1e-6)
    return seq_mean, seq_std, static_mean, static_std


def compute_target_normalization(ds: MPCDataset, indices: Iterable[int], target_mode: str) -> Tuple[np.ndarray, np.ndarray]:
    logs = []
    for idx in indices:
        item = ds[int(idx)]
        if target_mode == "qr_log_diag":
            target_log = torch.cat([
                safe_log10_diag(item["Q0"]),
                safe_log10_diag(item["R0"]),
            ]).numpy()
        else:
            target_log = torch.cat([
                safe_log10_diag(item["P0"]),
                safe_log10_diag(item["Q0"]),
                safe_log10_diag(item["R0"]),
            ]).numpy()
        logs.append(target_log)
    values = np.stack(logs, axis=0).astype(np.float32)
    mean = values.mean(axis=0).astype(np.float32)
    std = np.maximum(values.std(axis=0).astype(np.float32), 1e-6)
    return mean, std


def split_indices(n_samples: int, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    n_val = max(1, int(round(val_frac * n_samples)))
    return indices[n_val:], indices[:n_val]


def batch_losses(model: nn.Module, batch: Dict[str, torch.Tensor], device: torch.device,
                 weights: Dict[str, float]) -> Tuple[torch.Tensor, Dict[str, float]]:
    tokens_seq = batch["tokens_seq"].to(device)
    tokens_static = batch["tokens_static"].to(device)
    P0 = batch["P0"].to(device)
    Q0 = batch["Q0"].to(device)
    R0 = batch["R0"].to(device)
    A = batch["A"].to(device)
    B = batch["B"].to(device)
    u = batch["u"].to(device)

    if getattr(model, "predict_qr_log_diag", False):
        Qp, Rp, u_hat = model(tokens_seq, tokens_static)
        target_log = torch.cat([safe_log10_diag(Q0), safe_log10_diag(R0)], dim=-1)
        pred_log = torch.cat([Qp, Rp], dim=-1)

        target_mean = weights.get("target_mean")
        target_std = weights.get("target_std")
        if target_mean is not None and target_std is not None:
            target_log_norm = (target_log - target_mean) / target_std
            pred_log_for_metrics = pred_log * target_std + target_mean
            n = Qp.shape[-1]
            m = Rp.shape[-1]
            Qp_log = pred_log_for_metrics[:, :n]
            Rp_log = pred_log_for_metrics[:, n:n + m]
            loss_q = regression_loss(Qp, target_log_norm[:, :n], weights, "q")
            loss_r = regression_loss(Rp, target_log_norm[:, n:n + m], weights, "r")
        else:
            Qp_log, Rp_log = Qp, Rp
            loss_q = regression_loss(Qp_log, safe_log10_diag(Q0), weights, "q")
            loss_r = regression_loss(Rp_log, safe_log10_diag(R0), weights, "r")

        Q_ric = diag_matrix_from_log(Qp_log)
        R_ric = diag_matrix_from_log(Rp_log)
        P_ric = dare_iterate(A, B, Q_ric, R_ric, n_iter=int(weights.get("dare_iters", 80)))
        loss_p = frob_rel(P_ric, P0)
        metric_p = loss_p
        metric_q = rel_diag_from_log(Qp_log, Q0)
        metric_r = rel_diag_from_log(Rp_log, R0)
    else:
        Pp, Qp, Rp, u_hat = model(tokens_seq, tokens_static)

    if getattr(model, "predict_log_diag", False):
        target_log = torch.cat([safe_log10_diag(P0), safe_log10_diag(Q0), safe_log10_diag(R0)], dim=-1)
        pred_log = torch.cat([Pp, Qp, Rp], dim=-1)

        target_mean = weights.get("target_mean")
        target_std = weights.get("target_std")
        if target_mean is not None and target_std is not None:
            target_log_norm = (target_log - target_mean) / target_std
            pred_log_for_metrics = pred_log * target_std + target_mean
            n = Pp.shape[-1]
            m = Rp.shape[-1]
            Pp_log = pred_log_for_metrics[:, :n]
            Qp_log = pred_log_for_metrics[:, n:n + n]
            Rp_log = pred_log_for_metrics[:, n + n:n + n + m]
            loss_p = regression_loss(Pp, target_log_norm[:, :n], weights, "p")
            loss_q = regression_loss(Qp, target_log_norm[:, n:n + n], weights, "q")
            loss_r = regression_loss(Rp, target_log_norm[:, n + n:n + n + m], weights, "r")
        else:
            Pp_log, Qp_log, Rp_log = Pp, Qp, Rp
            loss_p = regression_loss(Pp_log, safe_log10_diag(P0), weights, "p")
            loss_q = regression_loss(Qp_log, safe_log10_diag(Q0), weights, "q")
            loss_r = regression_loss(Rp_log, safe_log10_diag(R0), weights, "r")

        metric_p = rel_diag_from_log(Pp_log, P0)
        metric_q = rel_diag_from_log(Qp_log, Q0)
        metric_r = rel_diag_from_log(Rp_log, R0)
        P_ric = diag_matrix_from_log(Pp_log)
        Q_ric = diag_matrix_from_log(Qp_log)
        R_ric = diag_matrix_from_log(Rp_log)
    elif not getattr(model, "predict_qr_log_diag", False):
        loss_p = frob_rel(Pp, P0)
        loss_q = frob_rel(Qp, Q0)
        loss_r = frob_rel(Rp, R0)
        metric_p = loss_p
        metric_q = loss_q
        metric_r = loss_r
        P_ric, Q_ric, R_ric = Pp, Qp, Rp

    loss_u = mse(u_hat, u)
    loss_ric = riccati_residual(P_ric, Q_ric, R_ric, A, B)
    loss = (
        weights["p"] * loss_p
        + weights["q"] * loss_q
        + weights["r"] * loss_r
        + weights["u"] * loss_u
        + weights["ric"] * loss_ric
    )

    metrics = {
        "loss": float(loss.detach()),
        "loss_p": float(loss_p.detach()),
        "loss_q": float(loss_q.detach()),
        "loss_r": float(loss_r.detach()),
        "rel_p": float(metric_p.detach()),
        "rel_q": float(metric_q.detach()),
        "rel_r": float(metric_r.detach()),
        "loss_u": float(loss_u.detach()),
        "loss_ric": float(loss_ric.detach()),
    }
    return loss, metrics


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
              weights: Dict[str, float], optimizer=None, grad_clip: float = 1.0,
              desc: str = "") -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: Dict[str, float] = {}
    n_batches = 0

    pbar = tqdm(loader, desc=desc, leave=False, bar_format="{l_bar}{bar:30}{r_bar}")
    for batch in pbar:
        with torch.set_grad_enabled(training):
            loss, metrics = batch_losses(model, batch, device, weights)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip and grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        n_batches += 1
        pbar.set_postfix(loss=f"{totals['loss']/n_batches:.3f}")

    return {key: value / max(n_batches, 1) for key, value in totals.items()}


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    p = argparse.ArgumentParser(description="Train the Transformer PQR model on the DC motor dataset.")
    p.add_argument("--data_root", type=str, default="MPC_dataset/mpc_qr_dataset/mpc_pqr_dataset_realfit_3000_spec_midanchor")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--weight_decay", type=float, default=0.03)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=default_device())
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--arch", choices=["transformer", "patchtst"], default="transformer")
    p.add_argument("--seq_pool", choices=["mean", "attention"], default="attention")
    p.add_argument("--patch_len", type=int, default=12)
    p.add_argument("--patch_stride", type=int, default=6)
    p.add_argument("--mamba_layers", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--tx_layers", type=int, default=2)
    p.add_argument("--tx_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--target_mode", choices=["full_psd", "log_diag", "qr_log_diag", "motor_struct"], default="log_diag")
    p.add_argument("--target_scale", choices=["none", "trace", "r"], default="none")
    p.add_argument("--target_loss", choices=["mse", "huber"], default="huber")
    p.add_argument("--target_loss_p", choices=["mse", "huber"], default=None)
    p.add_argument("--target_loss_q", choices=["mse", "huber"], default=None)
    p.add_argument("--target_loss_r", choices=["mse", "huber"], default="mse")
    p.add_argument("--huber_beta", type=float, default=0.25)
    p.add_argument("--w_p", type=float, default=2.0)
    p.add_argument("--w_q", type=float, default=2.0)
    p.add_argument("--w_r", type=float, default=2.0)
    p.add_argument("--w_u", type=float, default=1e-4)
    p.add_argument("--w_ric", type=float, default=0.0)
    p.add_argument("--dare_iters", type=int, default=80)
    p.add_argument("--out_dir", type=str, default="runs/paper_model")
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--init_checkpoint", type=str, default=None,
                   help="Load only model weights from a checkpoint, then start a fresh training run.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger = setup_logging(args.out_dir)
    device = torch.device(args.device)

    ds = MPCDataset(args.data_root, drop_duplicate_seq_feature=True, target_scale=args.target_scale)
    manifest = get_manifest(args.data_root, target_scale=args.target_scale)
    train_idx, val_idx = split_indices(len(ds), args.val_frac, args.seed)
    stats = compute_normalization(ds, train_idx)
    target_stats = compute_target_normalization(ds, train_idx, args.target_mode)
    ds.set_normalization_stats(*stats)

    train_loader = DataLoader(
        Subset(ds, train_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        Subset(ds, val_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    logger.info(f"Device: {device}")
    logger.info(f"Dataset: {args.data_root}")
    logger.info(f"Architecture: {args.arch}")
    logger.info(f"Target scale: {args.target_scale}")
    logger.info(f"Manifest: {manifest}")
    logger.info(f"Split: train={len(train_idx)} val={len(val_idx)}")

    model = HybridControllerModel(
        d_in=manifest["d_seq"],
        d_static=manifest["d_static"],
        d_model=args.d_model,
        n=manifest["n"],
        m=manifest["m"],
        mamba_layers=args.mamba_layers,
        tx_layers=args.tx_layers,
        tx_heads=args.tx_heads,
        dropout=args.dropout,
        eps=args.eps,
        predict_u=True,
        predict_log_diag=(args.target_mode == "log_diag"),
        predict_qr_log_diag=(args.target_mode == "qr_log_diag"),
        predict_motor_struct=(args.target_mode == "motor_struct"),
        arch=args.arch,
        seq_pool=args.seq_pool,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
    ).to(device)

    if args.target_mode == "motor_struct":
        with torch.no_grad():
            bias = model.motor_struct_head.net[-1].bias
            bias.zero_()
            log_means = torch.tensor(target_stats[0], dtype=bias.dtype, device=bias.device)
            # target_stats layout for n=2,m=1: [log P11, log P22, log Q11, log Q22, log R11].
            if log_means.numel() >= 5:
                bias[0] = log_means[0]
                bias[1] = log_means[1]
                bias[3] = log_means[2]
                bias[4] = log_means[3]
                bias[6] = log_means[4]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(epoch: int) -> float:
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(1, args.warmup_epochs)
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return args.min_lr / args.lr + (1 - args.min_lr / args.lr) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    weights = {"p": args.w_p, "q": args.w_q, "r": args.w_r, "u": args.w_u, "ric": args.w_ric}
    weights["target_loss"] = args.target_loss
    weights["target_loss_p"] = args.target_loss_p
    weights["target_loss_q"] = args.target_loss_q
    weights["target_loss_r"] = args.target_loss_r
    weights["huber_beta"] = args.huber_beta
    if args.target_mode in {"log_diag", "qr_log_diag"}:
        weights["target_mean"] = torch.tensor(target_stats[0], dtype=torch.float32, device=device)
        weights["target_std"] = torch.tensor(target_stats[1], dtype=torch.float32, device=device)
    weights["dare_iters"] = args.dare_iters
    start_epoch = 1
    best_val = float("inf")
    history = []

    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        logger.info(f"Initialized model weights from {args.init_checkpoint}")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["opt_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt.get("best_val", best_val))
        history = ckpt.get("history", [])
        for _ in range(start_epoch - 1):
            scheduler.step()

    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "manifest": manifest}, f, indent=2)
    np.savez(os.path.join(args.out_dir, "normalization.npz"),
             seq_mean=stats[0], seq_std=stats[1],
             static_mean=stats[2], static_std=stats[3],
             target_mean=target_stats[0], target_std=target_stats[1])

    n_params = sum(param.numel() for param in model.parameters())
    logger.info(f"Model params: {n_params / 1e6:.2f}M")
    logger.info(f"Loss weights: p={args.w_p}, q={args.w_q}, r={args.w_r}, u={args.w_u}, ric={args.w_ric}")
    logger.info(
        "Target loss: "
        f"default={args.target_loss}, "
        f"p={args.target_loss_p or args.target_loss}, "
        f"q={args.target_loss_q or args.target_loss}, "
        f"r={args.target_loss_r or args.target_loss}, "
        f"huber_beta={args.huber_beta}"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_metrics = run_epoch(
            model, train_loader, device, weights, optimizer, args.grad_clip,
            desc=f"Epoch {epoch:03d}/{args.epochs}",
        )
        val_metrics = run_epoch(model, val_loader, device, weights, desc="val")
        scheduler.step()

        is_best = val_metrics["loss"] < best_val
        if is_best:
            best_val = val_metrics["loss"]

        entry = {
            "epoch": epoch,
            "lr": lr,
            "time": time.time() - t0,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(entry)
        with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        logger.info(
            f"[{epoch:03d}/{args.epochs}] "
            f"train={train_metrics['loss']:.4f} "
            f"val={val_metrics['loss']:.4f} "
            f"rel_p/q/r={val_metrics['rel_p']:.3f}/{val_metrics['rel_q']:.3f}/{val_metrics['rel_r']:.3f} "
            f"log_p/q/r={val_metrics['loss_p']:.3f}/{val_metrics['loss_q']:.3f}/{val_metrics['loss_r']:.3f} "
            f"u={val_metrics['loss_u']:.3f} ric={val_metrics['loss_ric']:.3f} "
            f"lr={lr:.1e}{' *BEST*' if is_best else ''}"
        )

        if is_best or epoch % args.save_every == 0:
            name = "best.pt" if is_best else f"ckpt_epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "opt_state": optimizer.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                    "manifest": manifest,
                    "best_val": best_val,
                    "history": history,
                },
                os.path.join(args.out_dir, name),
            )

    logger.info(f"Done. Best val loss = {best_val:.4f}")


if __name__ == "__main__":
    main()
