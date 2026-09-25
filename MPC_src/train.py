"""
Training script for the Hybrid GRU-SSM + Transformer LQR controller model.
Fixed-dimension setup (no padding and no masks).
"""

import os
import sys
import math
import json
import time
import logging
import argparse

# Allow running from project root or from MPC_src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LQRDataset, get_num_systems, get_manifest
from model import HybridControllerModel


def setup_logging(out_dir):
    """Log to both terminal and file (train.log in out_dir)."""
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train.log")

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def dims_from_manifest(manifest):
    if "n" in manifest and "m" in manifest:
        return int(manifest["n"]), int(manifest["m"])
    if "n_max" in manifest and "m_max" in manifest:
        return int(manifest["n_max"]), int(manifest["m_max"])
    raise KeyError("manifest.json must contain either (n,m) or (n_max,m_max)")


def frob_rel(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean Frobenius relative error over batch."""
    num = torch.linalg.norm(a - b, ord="fro", dim=(-2, -1))
    den = torch.linalg.norm(b, ord="fro", dim=(-2, -1)) + eps
    return (num / den).mean()


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean squared error over all elements."""
    return ((a - b) ** 2).mean()


def riccati_residual(P, Q, R, A, B, eps=1e-8):
    """Compute Frobenius norm of DARE residual: P - Q - A'PA + A'PB(R+B'PB)^{-1}B'PA.
    Returns mean over batch."""
    # A'PA
    AtP = A.transpose(-1, -2) @ P
    AtPA = AtP @ A
    # B'PA
    BtPA = B.transpose(-1, -2) @ P @ A
    # R + B'PB
    BtPB = B.transpose(-1, -2) @ P @ B
    M = R + BtPB
    # Solve M^{-1} @ BtPA via Cholesky for numerical stability
    try:
        L = torch.linalg.cholesky(M + eps * torch.eye(M.shape[-1], device=M.device))
        MinvBtPA = torch.cholesky_solve(BtPA, L)
    except torch.linalg.LinAlgError:
        MinvBtPA = torch.linalg.solve(M + eps * torch.eye(M.shape[-1], device=M.device), BtPA)
    # Residual: P - Q - A'PA + A'PB M^{-1} B'PA
    residual = P - Q - AtPA + BtPA.transpose(-1, -2) @ MinvBtPA
    return torch.linalg.norm(residual, ord="fro", dim=(-2, -1)).mean()


@torch.no_grad()
def evaluate(model, loader, device, w_p, w_q, w_r, w_u, w_ric):
    """Evaluation on validation set."""
    model.eval()
    total_loss = 0.0
    total_p = 0.0
    total_q = 0.0
    total_r = 0.0
    total_u = 0.0
    total_ric = 0.0
    n_batches = 0

    for batch in loader:
        tokens = batch["tokens"].to(device)
        P0 = batch["P0"].to(device)
        Q0 = batch["Q0"].to(device)
        R0 = batch["R0"].to(device)
        A = batch["A"].to(device)
        B = batch["B"].to(device)
        u = batch["u"].to(device)
        dare_valid = batch["dare_valid"].to(device)

        Pp, Qp, Rp, u_hat = model(tokens)

        loss_q = frob_rel(Qp, Q0)
        loss_r = frob_rel(Rp, R0)
        loss_u = mse(u_hat, u)
        loss_ric = riccati_residual(Pp, Qp, Rp, A, B)

        # P loss only where DARE was solvable
        if dare_valid.sum() > 0:
            mask = dare_valid.bool()
            loss_p = frob_rel(Pp[mask], P0[mask])
        else:
            loss_p = torch.tensor(0.0, device=device)

        loss = w_p * loss_p + w_q * loss_q + w_r * loss_r + w_u * loss_u + w_ric * loss_ric

        total_loss += float(loss)
        total_p += float(loss_p)
        total_q += float(loss_q)
        total_r += float(loss_r)
        total_u += float(loss_u)
        total_ric += float(loss_ric)
        n_batches += 1

    if n_batches == 0:
        return {"loss": math.nan, "loss_p": math.nan, "loss_q": math.nan,
                "loss_r": math.nan, "loss_u": math.nan, "loss_ric": math.nan}

    return {
        "loss": total_loss / n_batches,
        "loss_p": total_p / n_batches,
        "loss_q": total_q / n_batches,
        "loss_r": total_r / n_batches,
        "loss_u": total_u / n_batches,
        "loss_ric": total_ric / n_batches,
    }


def train_one_epoch(model, loader, optimizer, scheduler, device, w_q, w_r, w_u, grad_clip, epoch, epochs):
    """Train one epoch (fixed dims, unmasked losses)."""
    model.train()
    total_loss = 0.0
    total_q = 0.0
    total_r = 0.0
    total_u = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d}/{epochs}", leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")

    for batch in pbar:
        tokens = batch["tokens"].to(device)
        Q0 = batch["Q0"].to(device)
        R0 = batch["R0"].to(device)
        u = batch["u"].to(device)

        Qp, Rp, u_hat = model(tokens)

        loss_q = frob_rel(Qp, Q0)
        loss_r = frob_rel(Rp, R0)
        loss_u = mse(u_hat, u)
        loss = w_q * loss_q + w_r * loss_r + w_u * loss_u

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += float(loss)
        total_q += float(loss_q)
        total_r += float(loss_r)
        total_u += float(loss_u)
        n_batches += 1

        pbar.set_postfix(
            loss=f"{total_loss/n_batches:.3f}",
            q=f"{total_q/n_batches:.3f}",
            r=f"{total_r/n_batches:.3f}",
            u=f"{total_u/n_batches:.2f}",
            lr=f"{optimizer.param_groups[0]['lr']:.1e}",
        )

    if scheduler is not None:
        scheduler.step()

    return {
        "loss": total_loss / max(n_batches, 1),
        "loss_q": total_q / max(n_batches, 1),
        "loss_r": total_r / max(n_batches, 1),
        "loss_u": total_u / max(n_batches, 1),
    }


def main():
    p = argparse.ArgumentParser()

    # Data
    p.add_argument("--data_root", type=str, default="MPC_dataset/synthetic_lqr_data")
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)

    # Model
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--mamba_layers", type=int, default=4)
    p.add_argument("--tx_layers", type=int, default=2)
    p.add_argument("--tx_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--eps", type=float, default=1e-3)

    # Train
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    def _default_device():
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    p.add_argument("--device", type=str, default=_default_device())

    # Loss weights
    p.add_argument("--w_q", type=float, default=5.0)
    p.add_argument("--w_r", type=float, default=5.0)
    p.add_argument("--w_u", type=float, default=1e-3)

    # Checkpointing
    p.add_argument("--out_dir", type=str, default="runs/run1")
    p.add_argument("--save_every", type=int, default=5)

    # Resume
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    args = p.parse_args()

    manifest = get_manifest(args.data_root)
    n, m = dims_from_manifest(manifest)
    d_in = n + m + 2  # y(n) + u_prev(m) + dt(1) + sat(1)

    args.n = n
    args.m = m
    args.d_in = d_in

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    logger = setup_logging(args.out_dir)

    device = torch.device(args.device)
    logger.info(f"Device: {device}")
    logger.info(f"Dataset: n={n}, m={m}, d_in={d_in}")

    # Dataset split by system ID
    num_systems = get_num_systems(args.data_root)
    n_val_sys = max(1, int(args.val_frac * num_systems))
    n_train_sys = num_systems - n_val_sys

    rng = np.random.RandomState(args.seed)
    all_sys_ids = list(range(num_systems))
    rng.shuffle(all_sys_ids)
    train_sys = set(all_sys_ids[:n_train_sys])
    val_sys = set(all_sys_ids[n_train_sys:])

    train_ds = LQRDataset(root_dir=args.data_root, sys_ids=train_sys)
    val_ds = LQRDataset(root_dir=args.data_root, sys_ids=val_sys)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    logger.info(f"Split by system: {n_train_sys} train systems ({len(train_ds)} episodes), "
                f"{n_val_sys} val systems ({len(val_ds)} episodes)")

    model = HybridControllerModel(
        d_in=d_in,
        d_model=args.d_model,
        n=n,
        m=m,
        mamba_layers=args.mamba_layers,
        tx_layers=args.tx_layers,
        tx_heads=args.tx_heads,
        dropout=args.dropout,
        eps=args.eps,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params/1e6:.2f}M parameters (d_model={args.d_model}, "
                f"mamba={args.mamba_layers}, tx={args.tx_layers})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return args.min_lr / args.lr + (1 - args.min_lr / args.lr) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 1
    best_val = float("inf")
    history = []

    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["opt_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        history = ckpt.get("history", [])
        for _ in range(start_epoch - 1):
            scheduler.step()
        logger.info(f"Resumed at epoch {start_epoch}, best_val={best_val:.4f}")

    with open(os.path.join(args.out_dir, "config.txt"), "w") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")

    logger.info(f"Loss weights: w_q={args.w_q}, w_r={args.w_r}, w_u={args.w_u}")
    logger.info(f"LR schedule: warmup={args.warmup_epochs} epochs, then cosine -> {args.min_lr}")
    logger.info(f"Training for {args.epochs} epochs (starting at {start_epoch})")
    logger.info("=" * 80)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            w_q=args.w_q, w_r=args.w_r, w_u=args.w_u,
            grad_clip=args.grad_clip, epoch=epoch, epochs=args.epochs,
        )

        val_metrics = evaluate(
            model, val_loader, device,
            w_q=args.w_q, w_r=args.w_r, w_u=args.w_u,
        )

        dt = time.time() - t0

        is_best = val_metrics["loss"] < best_val
        best_marker = " *BEST*" if is_best else ""

        logger.info(
            f"[{epoch:03d}/{args.epochs}] "
            f"train={train_metrics['loss']:.4f} (q={train_metrics['loss_q']:.4f} r={train_metrics['loss_r']:.4f} u={train_metrics['loss_u']:.2f}) | "
            f"val={val_metrics['loss']:.4f} (q={val_metrics['loss_q']:.4f} r={val_metrics['loss_r']:.4f} u={val_metrics['loss_u']:.2f}) | "
            f"lr={current_lr:.1e} | {dt:.0f}s{best_marker}"
        )

        entry = {
            "epoch": epoch,
            "lr": current_lr,
            "time": dt,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(entry)

        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.out_dir, f"ckpt_epoch_{epoch:03d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "opt_state": optimizer.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                    "best_val": best_val,
                    "history": history,
                },
                ckpt_path,
            )

        if is_best:
            best_val = val_metrics["loss"]
            best_path = os.path.join(args.out_dir, "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "opt_state": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "args": vars(args),
                    "best_val": best_val,
                    "history": history,
                },
                best_path,
            )

    logger.info("=" * 80)
    logger.info(f"Done. Best val loss = {best_val:.4f}")
    logger.info(f"Outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
