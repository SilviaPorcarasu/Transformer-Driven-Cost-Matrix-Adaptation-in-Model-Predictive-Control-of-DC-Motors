"""
Evaluation script for the Hybrid Mamba+Transformer LQR model.
Fixed-dimension setup (no padding and no masks).
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import LQRDataset, get_num_systems, get_manifest
from model import HybridControllerModel


def dims_from_manifest(manifest):
    if "n" in manifest and "m" in manifest:
        return int(manifest["n"]), int(manifest["m"])
    if "n_max" in manifest and "m_max" in manifest:
        return int(manifest["n_max"]), int(manifest["m_max"])
    raise KeyError("manifest.json must contain either (n,m) or (n_max,m_max)")


def frob_rel_per_sample(a, b, eps=1e-8):
    num = torch.linalg.norm(a - b, ord="fro", dim=(-2, -1))
    den = torch.linalg.norm(b, ord="fro", dim=(-2, -1)) + eps
    return num / den


def mse_per_sample(a, b):
    return ((a - b) ** 2).mean(dim=tuple(range(1, a.ndim)))


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run model on all batches, return collected results."""
    model.eval()
    results = {
        "Q_pred": [], "Q0": [],
        "R_pred": [], "R0": [],
        "u_hat": [], "u": [],
        "tokens": [],
    }
    for batch in loader:
        tokens = batch["tokens"].to(device)
        Qp, Rp, u_hat = model(tokens)

        results["Q_pred"].append(Qp.cpu())
        results["Q0"].append(batch["Q0"])
        results["R_pred"].append(Rp.cpu())
        results["R0"].append(batch["R0"])
        results["u_hat"].append(u_hat.cpu())
        results["u"].append(batch["u"])
        results["tokens"].append(batch["tokens"])

    return {k: torch.cat(v, dim=0) for k, v in results.items()}


def compute_metrics(results, w_q=1.0, w_r=1.0, w_u=1e-3):
    """Compute per-sample and aggregate metrics."""
    q_err = frob_rel_per_sample(results["Q_pred"], results["Q0"])
    r_err = frob_rel_per_sample(results["R_pred"], results["R0"])
    u_err = mse_per_sample(results["u_hat"], results["u"])

    total_loss = w_q * q_err + w_r * r_err + w_u * u_err

    def stats(t):
        t_np = t.numpy()
        pcts = np.percentile(t_np, [25, 50, 75, 95])
        return {
            "mean": float(t.mean()),
            "std": float(t.std()),
            "p25": float(pcts[0]),
            "median": float(pcts[1]),
            "p75": float(pcts[2]),
            "p95": float(pcts[3]),
        }

    return {
        "q_err": q_err,
        "r_err": r_err,
        "u_err": u_err,
        "total_loss": total_loss,
        "stats_q": stats(q_err),
        "stats_r": stats(r_err),
        "stats_u": stats(u_err),
        "stats_total": stats(total_loss),
    }


def plot_matrix_comparison(preds, targets, name, out_dir, num_samples=6):
    """Heatmap comparison: ground truth vs predicted vs error."""
    n = min(num_samples, preds.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(10, 3 * n))
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        gt = targets[i].numpy()
        pr = preds[i].numpy()
        err = np.abs(pr - gt)

        fre = float(frob_rel_per_sample(preds[i:i+1], targets[i:i+1]).item())

        vmax = max(np.abs(gt).max(), np.abs(pr).max())
        vmin = -vmax if gt.min() < 0 else 0

        axes[i, 0].imshow(gt, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f"GT {name}")
        for (r, c), v in np.ndenumerate(gt):
            axes[i, 0].text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7)

        axes[i, 1].imshow(pr, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f"Pred {name} (FRE={fre:.3f})")
        for (r, c), v in np.ndenumerate(pr):
            axes[i, 1].text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7)

        im = axes[i, 2].imshow(err, cmap="Reds")
        axes[i, 2].set_title("|Error|")
        for (r, c), v in np.ndenumerate(err):
            axes[i, 2].text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{name} Matrix Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name.lower()}_matrix_comparison.png"), dpi=150)
    plt.close(fig)


def plot_control_trajectories(u_hat, u, out_dir, num_samples=6):
    """Plot predicted vs ground truth control actions over time."""
    n = min(num_samples, u_hat.shape[0])
    m = u.shape[-1]

    fig, axes = plt.subplots(n, m, figsize=(6 * m, 3 * n))
    if n == 1:
        axes = axes[None, :]
    if m == 1:
        axes = axes[:, None]

    for i in range(n):
        sample_mse = float(((u_hat[i] - u[i]) ** 2).mean())
        for j in range(m):
            ax = axes[i, j]
            t_axis = np.arange(u.shape[1])
            ax.plot(t_axis, u[i, :, j].numpy(), label="GT", alpha=0.8, linewidth=1)
            ax.plot(t_axis, u_hat[i, :, j].numpy(), label="Pred", alpha=0.8,
                    linewidth=1, linestyle="--")
            if i == 0:
                ax.set_title(f"u[{j}]")
            if j == 0:
                ax.set_ylabel(f"Ep {i}\nMSE={sample_mse:.4f}")
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Control Trajectory Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "control_trajectories.png"), dpi=150)
    plt.close(fig)


def plot_loss_distributions(metrics, out_dir):
    """Histograms of per-sample errors."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, data, name, color in zip(
        axes,
        [metrics["q_err"].numpy(), metrics["r_err"].numpy(), metrics["u_err"].numpy()],
        ["Q Frobenius Rel. Error", "R Frobenius Rel. Error", "u MSE"],
        ["steelblue", "coral", "seagreen"],
    ):
        ax.hist(data, bins=50, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(np.median(data), color="black", linestyle="--",
                   label=f"median={np.median(data):.4f}")
        ax.set_title(name)
        ax.set_xlabel("Error")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Per-Sample Error Distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_distributions.png"), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Evaluate trained hybrid controller model")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    p.add_argument("--data_root", type=str, default="MPC_dataset/synthetic_lqr_data")
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_samples", type=int, default=6)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    device = torch.device(args.device)
    print(f"[device] {device}")

    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(args.checkpoint), "eval")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[checkpoint] {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt["args"]

    manifest = get_manifest(args.data_root)
    n_manifest, m_manifest = dims_from_manifest(manifest)
    n = int(saved_args.get("n", saved_args.get("n_max", n_manifest)))
    m = int(saved_args.get("m", saved_args.get("m_max", m_manifest)))

    model = HybridControllerModel(
        d_in=saved_args["d_in"],
        d_model=saved_args["d_model"],
        n=n,
        m=m,
        mamba_layers=saved_args["mamba_layers"],
        tx_layers=saved_args["tx_layers"],
        tx_heads=saved_args["tx_heads"],
        dropout=saved_args["dropout"],
        eps=saved_args["eps"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[model] loaded epoch {ckpt.get('epoch', '?')}")

    num_systems = get_num_systems(args.data_root)
    val_frac = saved_args.get("val_frac", 0.05)
    n_val_sys = max(1, int(val_frac * num_systems))
    n_train_sys = num_systems - n_val_sys

    rng = np.random.RandomState(args.seed)
    all_sys_ids = list(range(num_systems))
    rng.shuffle(all_sys_ids)
    val_sys = set(all_sys_ids[n_train_sys:])

    val_ds = LQRDataset(root_dir=args.data_root, sys_ids=val_sys)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[data] val={len(val_ds)} episodes from {n_val_sys} systems")

    print("[eval] running inference on validation set...")
    results = collect_predictions(model, val_loader, device)
    print(f"[eval] collected {results['Q_pred'].shape[0]} predictions")

    w_q = saved_args.get("w_q", 1.0)
    w_r = saved_args.get("w_r", 1.0)
    w_u = saved_args.get("w_u", 1e-3)
    metrics = compute_metrics(results, w_q=w_q, w_r=w_r, w_u=w_u)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for name, key in [("Q (Frob Rel)", "stats_q"), ("R (Frob Rel)", "stats_r"),
                      ("u (MSE)", "stats_u"), ("Total Loss", "stats_total")]:
        s = metrics[key]
        print(f"  {name:20s}  mean={s['mean']:.4f}  median={s['median']:.4f}  p95={s['p95']:.4f}")
    print("=" * 60)

    print("\n[viz] generating plots...")
    torch.manual_seed(args.seed)
    n_vis = min(args.num_samples, results["Q_pred"].shape[0])
    vis_idx = torch.randperm(results["Q_pred"].shape[0])[:n_vis]

    plot_matrix_comparison(
        results["Q_pred"][vis_idx], results["Q0"][vis_idx], "Q", args.out_dir, n_vis)
    plot_matrix_comparison(
        results["R_pred"][vis_idx], results["R0"][vis_idx], "R", args.out_dir, n_vis)
    plot_control_trajectories(
        results["u_hat"][vis_idx], results["u"][vis_idx], args.out_dir, n_vis)
    plot_loss_distributions(metrics, args.out_dir)

    summary = {
        "checkpoint": args.checkpoint,
        "epoch": ckpt.get("epoch", None),
        "device": str(device),
        "num_val_episodes": int(results["Q_pred"].shape[0]),
        "loss_weights": {"w_q": w_q, "w_r": w_r, "w_u": w_u},
        "Q_frob_rel": metrics["stats_q"],
        "R_frob_rel": metrics["stats_r"],
        "u_mse": metrics["stats_u"],
        "total_loss": metrics["stats_total"],
    }
    with open(os.path.join(args.out_dir, "eval_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[done] results saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
