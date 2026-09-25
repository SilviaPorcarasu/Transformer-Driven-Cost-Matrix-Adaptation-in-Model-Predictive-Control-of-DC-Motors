"""
Generate publication-quality figures and a complete experiment report.
Usage:
    python3 MPC_src/generate_report.py --run_dir runs/run1
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from dataset import LQRDataset
from model import HybridControllerModel

# Publication style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def frob_rel(a, b, eps=1e-8):
    num = torch.linalg.norm(a - b, ord="fro", dim=(-2, -1))
    den = torch.linalg.norm(b, ord="fro", dim=(-2, -1)) + eps
    return num / den


def load_history(run_dir):
    path = os.path.join(run_dir, "history.json")
    with open(path) as f:
        return json.load(f)


def load_config(run_dir):
    cfg = {}
    path = os.path.join(run_dir, "config.txt")
    with open(path) as f:
        for line in f:
            k, v = line.strip().split(": ", 1)
            cfg[k] = v
    return cfg


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    results = {"Q_pred": [], "Q0": [], "R_pred": [], "R0": [],
               "u_hat": [], "u": [], "tokens": []}
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


# ── Figure 1: Training Curves ──────────────────────────────────────────────

def plot_training_curves(history, out_dir):
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    # Total loss
    ax = axes[0, 0]
    ax.plot(epochs, [h["train_loss"] for h in history], "o-", label="Train", color="#2196F3")
    ax.plot(epochs, [h["val_loss"] for h in history], "s-", label="Val", color="#F44336")
    ax.set_ylabel("Total Weighted Loss")
    ax.set_title("(a) Total Loss")
    ax.legend()

    # Q loss
    ax = axes[0, 1]
    ax.plot(epochs, [h["train_loss_q"] for h in history], "o-", label="Train", color="#2196F3")
    ax.plot(epochs, [h["val_loss_q"] for h in history], "s-", label="Val", color="#F44336")
    ax.set_ylabel("Frobenius Relative Error")
    ax.set_title("(b) Q Matrix Loss")
    ax.legend()

    # R loss
    ax = axes[1, 0]
    ax.plot(epochs, [h["train_loss_r"] for h in history], "o-", label="Train", color="#2196F3")
    ax.plot(epochs, [h["val_loss_r"] for h in history], "s-", label="Val", color="#F44336")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Frobenius Relative Error")
    ax.set_title("(c) R Matrix Loss")
    ax.legend()

    # u loss
    ax = axes[1, 1]
    ax.plot(epochs, [h["train_loss_u"] for h in history], "o-", label="Train", color="#2196F3")
    ax.plot(epochs, [h["val_loss_u"] for h in history], "s-", label="Val", color="#F44336")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("(d) Control Action Loss")
    ax.legend()

    fig.suptitle("Training and Validation Loss Curves", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_training_curves.png"))
    fig.savefig(os.path.join(out_dir, "fig1_training_curves.pdf"))
    plt.close(fig)
    print("  -> fig1_training_curves.png/.pdf")


# ── Figure 2: Error Distributions ──────────────────────────────────────────

def plot_error_distributions(results, out_dir):
    q_err = frob_rel(results["Q_pred"], results["Q0"]).numpy()
    r_err = frob_rel(results["R_pred"], results["R0"]).numpy()
    u_err = ((results["u_hat"] - results["u"]) ** 2).mean(dim=(1, 2)).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    for ax, data, title, color, xlabel in zip(
        axes,
        [q_err, r_err, u_err],
        ["(a) Q Frobenius Rel. Error", "(b) R Frobenius Rel. Error", "(c) Control MSE"],
        ["#2196F3", "#FF9800", "#4CAF50"],
        ["FRE", "FRE", "MSE"],
    ):
        ax.hist(data, bins=50, color=color, edgecolor="white", alpha=0.85)
        med = np.median(data)
        ax.axvline(med, color="black", linestyle="--", linewidth=1.5,
                   label=f"median = {med:.4f}")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.legend()

    fig.suptitle("Per-Episode Error Distributions on Validation Set", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_error_distributions.png"))
    fig.savefig(os.path.join(out_dir, "fig2_error_distributions.pdf"))
    plt.close(fig)
    print("  -> fig2_error_distributions.png/.pdf")


# ── Figure 3: Matrix Comparison ────────────────────────────────────────────

def plot_matrix_comparison(results, out_dir, num_samples=4):
    torch.manual_seed(42)
    idx = torch.randperm(results["Q_pred"].shape[0])[:num_samples]

    fig, axes = plt.subplots(num_samples, 6, figsize=(16, 3 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]

    for i, si in enumerate(idx):
        gt_q = results["Q0"][si].numpy()
        pr_q = results["Q_pred"][si].numpy()
        gt_r = results["R0"][si].numpy()
        pr_r = results["R_pred"][si].numpy()

        fre_q = float(frob_rel(results["Q_pred"][si:si+1], results["Q0"][si:si+1]))
        fre_r = float(frob_rel(results["R_pred"][si:si+1], results["R0"][si:si+1]))

        for j, (mat, title) in enumerate([
            (gt_q, "GT Q"), (pr_q, f"Pred Q\nFRE={fre_q:.3f}"), (np.abs(pr_q - gt_q), "|Error| Q"),
            (gt_r, "GT R"), (pr_r, f"Pred R\nFRE={fre_r:.3f}"), (np.abs(pr_r - gt_r), "|Error| R"),
        ]):
            ax = axes[i, j]
            cmap = "Reds" if "Error" in title else "RdBu_r"
            vmax = np.abs(mat).max() if "Error" in title else max(np.abs(mat).max(), 0.01)
            vmin = 0 if "Error" in title else -vmax
            im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
            for (r, c), v in np.ndenumerate(mat):
                ax.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=7)
            if i == 0:
                ax.set_title(title, fontsize=9)
            elif "Error" not in title:
                ax.set_title(title.split("\n")[-1], fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"Episode {int(si)}", fontsize=9)

    fig.suptitle("Q and R Matrix Predictions vs Ground Truth", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_matrix_comparison.png"))
    fig.savefig(os.path.join(out_dir, "fig3_matrix_comparison.pdf"))
    plt.close(fig)
    print("  -> fig3_matrix_comparison.png/.pdf")


# ── Figure 4: Control Trajectories ─────────────────────────────────────────

def plot_control_trajectories(results, out_dir, num_samples=4):
    torch.manual_seed(123)
    idx = torch.randperm(results["u"].shape[0])[:num_samples]
    m = results["u"].shape[-1]

    fig, axes = plt.subplots(num_samples, m, figsize=(6 * m, 2.5 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]

    for i, si in enumerate(idx):
        u_gt = results["u"][si].numpy()
        u_pr = results["u_hat"][si].numpy()
        ep_mse = float(((results["u_hat"][si] - results["u"][si]) ** 2).mean())
        T = u_gt.shape[0]
        t = np.arange(T)

        for j in range(m):
            ax = axes[i, j]
            ax.plot(t, u_gt[:, j], color="#2196F3", linewidth=1, alpha=0.8, label="Ground Truth")
            ax.plot(t, u_pr[:, j], color="#F44336", linewidth=1, alpha=0.8, linestyle="--", label="Predicted")
            if i == 0:
                ax.set_title(f"Control input u[{j}]", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"Ep {int(si)}\nMSE={ep_mse:.3f}", fontsize=9)
            if i == num_samples - 1:
                ax.set_xlabel("Timestep")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)

    fig.suptitle("Predicted vs Ground Truth Control Actions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_control_trajectories.png"))
    fig.savefig(os.path.join(out_dir, "fig4_control_trajectories.pdf"))
    plt.close(fig)
    print("  -> fig4_control_trajectories.png/.pdf")


# ── Figure 5: Diagonal Scatter ─────────────────────────────────────────────

def plot_diagonal_scatter(results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, preds, targets, name, color in [
        (axes[0], results["Q_pred"], results["Q0"], "Q", "#2196F3"),
        (axes[1], results["R_pred"], results["R0"], "R", "#FF9800"),
    ]:
        n_mat = preds.shape[-1]
        diag_pred = torch.stack([preds[:, i, i] for i in range(n_mat)], dim=-1).numpy().flatten()
        diag_gt = torch.stack([targets[:, i, i] for i in range(n_mat)], dim=-1).numpy().flatten()

        ax.scatter(diag_gt, diag_pred, alpha=0.3, s=10, color=color, edgecolors="none")
        lims = [min(diag_gt.min(), diag_pred.min()), max(diag_gt.max(), diag_pred.max())]
        ax.plot(lims, lims, "k--", linewidth=1, label="y = x (perfect)")
        ax.set_xlabel(f"Ground Truth {name} diagonal")
        ax.set_ylabel(f"Predicted {name} diagonal")
        ax.set_title(f"({'a' if name == 'Q' else 'b'}) {name} Matrix Diagonal Elements")
        ax.legend()
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle("Predicted vs Ground Truth Diagonal Elements", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig5_diagonal_scatter.png"))
    fig.savefig(os.path.join(out_dir, "fig5_diagonal_scatter.pdf"))
    plt.close(fig)
    print("  -> fig5_diagonal_scatter.png/.pdf")


# ── Experiment Summary (text) ──────────────────────────────────────────────

def write_experiment_summary(config, history, eval_results, results, run_dir, out_dir):
    q_err = frob_rel(results["Q_pred"], results["Q0"]).numpy()
    r_err = frob_rel(results["R_pred"], results["R0"]).numpy()
    u_err = ((results["u_hat"] - results["u"]) ** 2).mean(dim=(1, 2)).numpy()

    n_params = sum(p.numel() for p in torch.nn.Module().parameters()) if False else "see below"

    # Count params from checkpoint
    ckpt = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu", weights_only=False)
    n_params = sum(v.numel() for v in ckpt["model_state"].values())

    summary = f"""EXPERIMENT REPORT
{'='*70}

1. DATASET
   - Source: Synthetic LQR closed-loop trajectories
   - Systems: 200 random discrete-time LTI systems (n=4 states, m=2 inputs)
   - Episodes per system: 50 (with disturbances: step, sine, drift, impulse, none)
   - Trajectory length: T = 256 timesteps
   - Total episodes: {config.get('data_root', 'N/A')} -> 9,194 valid episodes
   - Train/Val split: 95%/5% (8,735 train / 459 val), seed=42
   - Token features (d_in=8): [y_t(4), u_{{t-1}}(2), dt(1), sat_flag(1)]

2. MODEL ARCHITECTURE
   - Type: Hybrid GRU-SSM + Transformer
   - Input projection: Linear(8 -> {config.get('d_model', '?')})
   - SSM backbone: {config.get('mamba_layers', '?')} GRU layers (d_model={config.get('d_model', '?')}, expand=2)
     with depthwise Conv1D + SiLU gating (Mamba-style block)
   - Bridge: LayerNorm + Linear({config.get('d_model', '?')} -> {config.get('d_model', '?')})
   - Transformer: {config.get('tx_layers', '?')} encoder layers, {config.get('tx_heads', '?')} heads, pre-norm, GELU
   - QR Head: MLP -> Cholesky L*L^T + eps*I (guarantees PSD)
   - Policy Head: MLP per timestep -> u_hat(t)
   - Total parameters: {n_params:,} ({n_params/1e6:.2f}M)

3. TRAINING CONFIGURATION
   - Optimizer: AdamW (lr={config.get('lr', '?')}, weight_decay={config.get('weight_decay', '?')})
   - Gradient clipping: {config.get('grad_clip', '?')}
   - Batch size: {config.get('batch_size', '?')}
   - Epochs: {config.get('epochs', '?')}
   - Loss: w_q * FRE(Q_pred, Q0) + w_r * FRE(R_pred, R0) + w_u * MSE(u_hat, u)
     where FRE = ||A-B||_F / ||B||_F (Frobenius Relative Error)
   - Loss weights: w_q={config.get('w_q', '?')}, w_r={config.get('w_r', '?')}, w_u={config.get('w_u', '?')}
   - Dropout: {config.get('dropout', '?')}
   - Seed: {config.get('seed', '?')}
   - Device: CPU (Apple Silicon M-series, no CUDA)

4. TRAINING RESULTS (per epoch)
{'   Epoch | Train Loss | Val Loss   | Val Q FRE  | Val R FRE  | Val u MSE'}
{'   ' + '-'*65}"""

    for h in history:
        summary += f"\n   {h['epoch']:5d} | {h['train_loss']:10.4f} | {h['val_loss']:10.4f} | {h['val_loss_q']:10.4f} | {h['val_loss_r']:10.4f} | {h['val_loss_u']:10.2f}"

    summary += f"""

5. EVALUATION RESULTS (best checkpoint, epoch {eval_results.get('epoch', '?')})
   Evaluated on {eval_results.get('num_val_episodes', '?')} validation episodes.

   Q Matrix (Frobenius Relative Error):
     Mean:   {q_err.mean():.4f}
     Median: {np.median(q_err):.4f}
     Std:    {q_err.std():.4f}
     P25:    {np.percentile(q_err, 25):.4f}
     P75:    {np.percentile(q_err, 75):.4f}
     P95:    {np.percentile(q_err, 95):.4f}

   R Matrix (Frobenius Relative Error):
     Mean:   {r_err.mean():.4f}
     Median: {np.median(r_err):.4f}
     Std:    {r_err.std():.4f}
     P25:    {np.percentile(r_err, 25):.4f}
     P75:    {np.percentile(r_err, 75):.4f}
     P95:    {np.percentile(r_err, 95):.4f}

   Control Actions (MSE):
     Mean:   {u_err.mean():.4f}
     Median: {np.median(u_err):.4f}
     Std:    {u_err.std():.4f}
     P25:    {np.percentile(u_err, 25):.4f}
     P75:    {np.percentile(u_err, 75):.4f}
     P95:    {np.percentile(u_err, 95):.4f}

6. FILES GENERATED
   - fig1_training_curves.png/.pdf    : Training & validation loss curves
   - fig2_error_distributions.png/.pdf: Per-episode error histograms
   - fig3_matrix_comparison.png/.pdf  : Q/R predicted vs ground truth heatmaps
   - fig4_control_trajectories.png/.pdf: u_hat vs u over time
   - fig5_diagonal_scatter.png/.pdf   : Predicted vs GT diagonal elements
   - experiment_report.txt            : This file
   - eval_results.json                : Machine-readable evaluation metrics

{'='*70}
"""

    report_path = os.path.join(out_dir, "experiment_report.txt")
    with open(report_path, "w") as f:
        f.write(summary)
    print(f"  -> experiment_report.txt")
    return summary


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--data_root", type=str, default="MPC_dataset/synthetic_lqr_data")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--num_samples", type=int, default=4)
    args = p.parse_args()

    run_dir = args.run_dir
    out_dir = os.path.join(run_dir, "report")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device)

    print(f"Loading run from {run_dir}...")
    config = load_config(run_dir)
    history = load_history(run_dir)

    eval_path = os.path.join(run_dir, "eval", "eval_results.json")
    with open(eval_path) as f:
        eval_results = json.load(f)

    # Load model + data
    ckpt = torch.load(os.path.join(run_dir, "best.pt"), map_location=device, weights_only=False)
    saved_args = ckpt["args"]

    model = HybridControllerModel(
        d_in=saved_args["d_in"], d_model=saved_args["d_model"],
        n=saved_args["n"], m=saved_args["m"],
        mamba_layers=saved_args["mamba_layers"], tx_layers=saved_args["tx_layers"],
        tx_heads=saved_args["tx_heads"], dropout=saved_args["dropout"],
        eps=saved_args["eps"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    ds = LQRDataset(root_dir=args.data_root)
    n_total = len(ds)
    val_frac = saved_args.get("val_frac", 0.05)
    n_val = max(1, int(val_frac * n_total))
    n_train = n_total - n_val
    _, val_ds = random_split(ds, [n_train, n_val],
                             generator=torch.Generator().manual_seed(saved_args.get("seed", 42)))
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    print("Running inference...")
    results = collect_predictions(model, val_loader, device)
    print(f"Collected {results['Q_pred'].shape[0]} predictions.\n")

    print("Generating figures:")
    plot_training_curves(history, out_dir)
    plot_error_distributions(results, out_dir)
    plot_matrix_comparison(results, out_dir, args.num_samples)
    plot_control_trajectories(results, out_dir, args.num_samples)
    plot_diagonal_scatter(results, out_dir)

    print("\nWriting experiment report:")
    summary = write_experiment_summary(config, history, eval_results, results, run_dir, out_dir)

    print(f"\nAll files saved to: {out_dir}/")
    print("\n" + summary)


if __name__ == "__main__":
    main()
