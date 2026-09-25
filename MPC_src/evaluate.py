"""
Evaluation script for the Hybrid Mamba+Transformer LQR model.
Predicts P, Q, R cost matrices only.
Includes: R² score, Riccati coherence, MinT projection.
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


# ── R² score ──
def r2_score_matrix(pred, target):
    """R² score for batched matrices. Flatten each sample, compute R² over elements."""
    pred_flat = pred.reshape(pred.shape[0], -1)
    tgt_flat = target.reshape(target.shape[0], -1)
    ss_res = ((tgt_flat - pred_flat) ** 2).sum(dim=-1)
    ss_tot = ((tgt_flat - tgt_flat.mean(dim=-1, keepdim=True)) ** 2).sum(dim=-1)
    r2 = 1.0 - ss_res / (ss_tot + 1e-8)
    return r2  # (B,)


# ── Riccati residual: P - Q - A'PA + A'PB(R+B'PB)^{-1}B'PA ──
def riccati_residual_per_sample(P, Q, R, A, B, eps=1e-8):
    """Frobenius norm of DARE residual per sample."""
    AtP = A.transpose(-1, -2) @ P
    AtPA = AtP @ A
    BtPA = B.transpose(-1, -2) @ P @ A
    BtPB = B.transpose(-1, -2) @ P @ B
    M = R + BtPB
    eye = eps * torch.eye(M.shape[-1], device=M.device)
    MinvBtPA = torch.linalg.solve(M + eye, BtPA)
    residual = P - Q - AtPA + BtPA.transpose(-1, -2) @ MinvBtPA
    return torch.linalg.norm(residual, ord="fro", dim=(-2, -1))  # (B,)


# ── DARE residual for a single sample ──
def _dare_residual_single(P, Q, R, A, B, eps=1e-8):
    """c(P,Q,R) = P - Q - A'PA + A'PB(R+B'PB)^{-1}B'PA.  Returns (n,n)."""
    AtPA = A.T @ P @ A
    BtPA = B.T @ P @ A
    BtPB = B.T @ P @ B
    M = R + BtPB + eps * np.eye(R.shape[0])
    MinvBtPA = np.linalg.solve(M, BtPA)
    return P - Q - AtPA + BtPA.T @ MinvBtPA


def _numerical_jacobian(P, Q, R, A, B, eps_fd=1e-5):
    """Numerical Jacobian dc/dθ where θ = [vec(P), vec(Q), vec(R)].

    Returns C of shape (n², 2n² + m²).
    """
    n = P.shape[0]
    m = R.shape[0]
    n2 = n * n
    m2 = m * m
    dim_theta = 2 * n2 + m2

    theta = np.concatenate([P.ravel(), Q.ravel(), R.ravel()])
    c0 = _dare_residual_single(P, Q, R, A, B).ravel()

    C = np.zeros((n2, dim_theta), dtype=np.float64)

    for i in range(dim_theta):
        theta_p = theta.copy()
        theta_p[i] += eps_fd
        Pi = theta_p[:n2].reshape(n, n)
        Qi = theta_p[n2:2 * n2].reshape(n, n)
        Ri = theta_p[2 * n2:].reshape(m, m)
        ci = _dare_residual_single(Pi, Qi, Ri, A, B).ravel()
        C[:, i] = (ci - c0) / eps_fd

    return C


def _project_psd(M, eps=1e-6):
    """Project a symmetric matrix onto the PSD cone (clip negative eigenvalues)."""
    M_sym = 0.5 * (M + M.T)
    eigvals, eigvecs = np.linalg.eigh(M_sym)
    eigvals = np.maximum(eigvals, eps)
    return (eigvecs * eigvals) @ eigvecs.T


# ── MinT projection with Lagrange multipliers ──
def mint_projection(P_pred, Q_pred, R_pred, A, B, W=None, n_iter=3, eps_fd=1e-5):
    """
    Linearizes the nonlinear DARE constraint c(θ) = 0 at the current estimate
    and applies the minimum-trace projection with Lagrange multipliers:

        θ̃ = θ̂ - W · C' · (C · W · C')⁻¹ · c(θ̂)

    where:
        θ = [vec(P), vec(Q), vec(R)]  — parameter vector
        c(θ) = vec(DARE residual)     — constraint violation (n²)
        C = dc/dθ                     — Jacobian at θ̂ (n² × dim_θ)
        W = covariance of prediction errors (default: I)

    Iterated n_iter times (Gauss-Newton on the nonlinear constraint).
    All three matrices P, Q, R are adjusted (not just P).

    Returns: P_proj, Q_proj, R_proj (all reconciled).
    """
    A_np = A.cpu().numpy().astype(np.float64)
    B_np = B.cpu().numpy().astype(np.float64)
    P_np = P_pred.cpu().numpy().astype(np.float64)
    Q_np = Q_pred.cpu().numpy().astype(np.float64)
    R_np = R_pred.cpu().numpy().astype(np.float64)

    batch_size = P_pred.shape[0]
    n = P_pred.shape[1]
    m = R_pred.shape[1]
    n2 = n * n
    m2 = m * m

    P_out = np.copy(P_np)
    Q_out = np.copy(Q_np)
    R_out = np.copy(R_np)

    for b in range(batch_size):
        Ab, Bb = A_np[b], B_np[b]
        Pb, Qb, Rb = P_out[b], Q_out[b], R_out[b]

        # Per-sample W (if provided) or identity
        if W is not None:
            Wb = W[b].cpu().numpy().astype(np.float64) if torch.is_tensor(W) else W
        else:
            Wb = None  # use identity

        for it in range(n_iter):
            # 1. Evaluate constraint violation
            c_vec = _dare_residual_single(Pb, Qb, Rb, Ab, Bb).ravel()

            if np.linalg.norm(c_vec) < 1e-10:
                break

            # 2. Compute Jacobian C = dc/dθ at current (P, Q, R)
            C = _numerical_jacobian(Pb, Qb, Rb, Ab, Bb, eps_fd=eps_fd)

            # 3. Projection: θ̃ = θ̂ - W·C'·(C·W·C')⁻¹·c(θ̂)
            theta = np.concatenate([Pb.ravel(), Qb.ravel(), Rb.ravel()])

            if Wb is None:
                # W = I: correction = C' · (C·C')⁻¹ · c
                CCtinv_c = np.linalg.solve(C @ C.T + 1e-8 * np.eye(n2), c_vec)
                correction = C.T @ CCtinv_c
            else:
                # General W: correction = W · C' · (C·W·C')⁻¹ · c
                CW = C @ Wb
                CWCt = CW @ C.T + 1e-8 * np.eye(n2)
                CWCtinv_c = np.linalg.solve(CWCt, c_vec)
                correction = Wb @ C.T @ CWCtinv_c

            theta_new = theta - correction

            # 4. Reshape back and project onto PSD cone
            Pb = _project_psd(theta_new[:n2].reshape(n, n))
            Qb = _project_psd(theta_new[n2:2 * n2].reshape(n, n))
            Rb = _project_psd(theta_new[2 * n2:].reshape(m, m))

        P_out[b] = Pb
        Q_out[b] = Qb
        R_out[b] = Rb

    return (
        torch.from_numpy(P_out).float().to(P_pred.device),
        torch.from_numpy(Q_out).float().to(Q_pred.device),
        torch.from_numpy(R_out).float().to(R_pred.device),
    )


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run model on all batches, return collected results."""
    model.eval()
    results = {
        "P_pred": [], "P0": [],
        "Q_pred": [], "Q0": [],
        "R_pred": [], "R0": [],
        "A": [], "B": [],
        "dare_valid": [],
    }
    for batch in loader:
        tokens = batch["tokens"].to(device)
        Pp, Qp, Rp = model(tokens)

        results["P_pred"].append(Pp.cpu())
        results["P0"].append(batch["P0"])
        results["Q_pred"].append(Qp.cpu())
        results["Q0"].append(batch["Q0"])
        results["R_pred"].append(Rp.cpu())
        results["R0"].append(batch["R0"])
        results["A"].append(batch["A"])
        results["B"].append(batch["B"])
        results["dare_valid"].append(batch["dare_valid"])

    return {k: torch.cat(v, dim=0) for k, v in results.items()}


def compute_metrics(results, run_mint=True):
    """Compute per-sample and aggregate metrics: FRE, R², Riccati, MinT."""

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

    # ── Frobenius relative errors ──
    q_err = frob_rel_per_sample(results["Q_pred"], results["Q0"])
    r_err = frob_rel_per_sample(results["R_pred"], results["R0"])

    dare_valid = results["dare_valid"].bool()

    if dare_valid.sum() > 0:
        p_err_valid = frob_rel_per_sample(
            results["P_pred"][dare_valid], results["P0"][dare_valid])
        p_err = torch.full((results["P_pred"].shape[0],), float("nan"))
        p_err[dare_valid] = p_err_valid
    else:
        p_err = torch.full((results["P_pred"].shape[0],), float("nan"))

    # ── R² scores ──
    r2_q = r2_score_matrix(results["Q_pred"], results["Q0"])
    r2_r = r2_score_matrix(results["R_pred"], results["R0"])
    if dare_valid.sum() > 0:
        r2_p_valid = r2_score_matrix(
            results["P_pred"][dare_valid], results["P0"][dare_valid])
        r2_p = torch.full((results["P_pred"].shape[0],), float("nan"))
        r2_p[dare_valid] = r2_p_valid
    else:
        r2_p = torch.full((results["P_pred"].shape[0],), float("nan"))

    # ── Riccati coherence residual ──
    ric_res = riccati_residual_per_sample(
        results["P_pred"], results["Q_pred"], results["R_pred"],
        results["A"], results["B"])

    # ── MinT projection ──
    mint_metrics = {}
    if run_mint:
        print("[eval] running MinT projection (reconciliation)...")
        P_proj, Q_proj, R_proj = mint_projection(
            results["P_pred"], results["Q_pred"], results["R_pred"],
            results["A"], results["B"])

        ric_res_after = riccati_residual_per_sample(
            P_proj, Q_proj, R_proj, results["A"], results["B"])

        q_err_after = frob_rel_per_sample(Q_proj, results["Q0"])
        r_err_after = frob_rel_per_sample(R_proj, results["R0"])
        r2_q_after = r2_score_matrix(Q_proj, results["Q0"])
        r2_r_after = r2_score_matrix(R_proj, results["R0"])

        if dare_valid.sum() > 0:
            p_err_after = frob_rel_per_sample(P_proj[dare_valid], results["P0"][dare_valid])
            r2_p_after = r2_score_matrix(P_proj[dare_valid], results["P0"][dare_valid])
        else:
            p_err_after = torch.tensor([float("nan")])
            r2_p_after = torch.tensor([float("nan")])

        mint_metrics = {
            "P_proj": P_proj, "Q_proj": Q_proj, "R_proj": R_proj,
            "ric_residual_after_mint": ric_res_after,
            "stats_ric_after_mint": stats(ric_res_after),
            "stats_p_after_mint": stats(p_err_after),
            "stats_q_after_mint": stats(q_err_after),
            "stats_r_after_mint": stats(r_err_after),
            "stats_r2_p_after_mint": stats(r2_p_after),
            "stats_r2_q_after_mint": stats(r2_q_after),
            "stats_r2_r_after_mint": stats(r2_r_after),
        }

    return {
        # Per-sample errors
        "p_err": p_err, "q_err": q_err, "r_err": r_err,
        # R² scores
        "r2_p": r2_p, "r2_q": r2_q, "r2_r": r2_r,
        # Riccati residual (before reconciliation)
        "ric_residual": ric_res,
        # Stats
        "stats_p": stats(p_err[dare_valid]) if dare_valid.sum() > 0 else {},
        "stats_q": stats(q_err),
        "stats_r": stats(r_err),
        "stats_r2_p": stats(r2_p[dare_valid]) if dare_valid.sum() > 0 else {},
        "stats_r2_q": stats(r2_q),
        "stats_r2_r": stats(r2_r),
        "stats_ric": stats(ric_res),
        # MinT results
        **mint_metrics,
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


def plot_loss_distributions(metrics, out_dir):
    """Histograms of per-sample errors for P, Q, R."""
    p_data = metrics["p_err"].numpy()
    p_valid = p_data[~np.isnan(p_data)]

    n_plots = 3 if len(p_valid) == 0 else 3
    plot_items = [
        (metrics["q_err"].numpy(), "Q Frobenius Rel. Error", "steelblue"),
        (metrics["r_err"].numpy(), "R Frobenius Rel. Error", "coral"),
    ]
    if len(p_valid) > 0:
        plot_items.insert(0, (p_valid, "P Frobenius Rel. Error", "mediumpurple"))

    n_plots = len(plot_items)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for ax, (data, name, color) in zip(axes, plot_items):
        ax.hist(data, bins=50, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(np.median(data), color="black", linestyle="--",
                   label=f"median={np.median(data):.4f}")
        ax.set_title(name)
        ax.set_xlabel("Error")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Per-Sample Error Distributions (P, Q, R)", fontsize=14)
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

    metrics = compute_metrics(results, run_mint=True)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print("\n  -- Frobenius Relative Error --")
    for name, key in [("P (Frob Rel)", "stats_p"), ("Q (Frob Rel)", "stats_q"),
                      ("R (Frob Rel)", "stats_r")]:
        s = metrics.get(key, {})
        if s:
            print(f"  {name:20s}  mean={s['mean']:.4f}  median={s['median']:.4f}  p95={s['p95']:.4f}")

    print("\n  -- R² Score --")
    for name, key in [("R²(P)", "stats_r2_p"), ("R²(Q)", "stats_r2_q"),
                      ("R²(R)", "stats_r2_r")]:
        s = metrics.get(key, {})
        if s:
            print(f"  {name:20s}  mean={s['mean']:.4f}  median={s['median']:.4f}")

    print("\n  -- Riccati Coherence --")
    s = metrics.get("stats_ric", {})
    if s:
        print(f"  {'Before MinT':20s}  mean={s['mean']:.4f}  median={s['median']:.4f}  p95={s['p95']:.4f}")
    s = metrics.get("stats_ric_after_mint", {})
    if s:
        print(f"  {'After MinT':20s}  mean={s['mean']:.4f}  median={s['median']:.4f}  p95={s['p95']:.4f}")

    if metrics.get("stats_p_after_mint"):
        print(f"\n  -- After MinT Projection (Frobenius Rel. Error) --")
        for name, key in [("P", "stats_p_after_mint"), ("Q", "stats_q_after_mint"),
                          ("R", "stats_r_after_mint")]:
            s = metrics.get(key, {})
            if s:
                print(f"  {name:20s}  mean={s['mean']:.4f}  median={s['median']:.4f}  p95={s['p95']:.4f}")

        print(f"\n  -- After MinT Projection (R² Score) --")
        for name, key in [("R²(P)", "stats_r2_p_after_mint"), ("R²(Q)", "stats_r2_q_after_mint"),
                          ("R²(R)", "stats_r2_r_after_mint")]:
            s = metrics.get(key, {})
            if s:
                print(f"  {name:20s}  mean={s['mean']:.4f}  median={s['median']:.4f}")

    print("=" * 60)

    print("\n[viz] generating plots...")
    torch.manual_seed(args.seed)
    n_vis = min(args.num_samples, results["Q_pred"].shape[0])
    vis_idx = torch.randperm(results["Q_pred"].shape[0])[:n_vis]

    dare_vis = results["dare_valid"][vis_idx].bool()
    if dare_vis.sum() > 0:
        valid_vis = vis_idx[dare_vis]
        plot_matrix_comparison(
            results["P_pred"][valid_vis], results["P0"][valid_vis], "P", args.out_dir,
            min(n_vis, int(dare_vis.sum())))

    plot_matrix_comparison(
        results["Q_pred"][vis_idx], results["Q0"][vis_idx], "Q", args.out_dir, n_vis)
    plot_matrix_comparison(
        results["R_pred"][vis_idx], results["R0"][vis_idx], "R", args.out_dir, n_vis)
    plot_loss_distributions(metrics, args.out_dir)

    summary = {
        "checkpoint": args.checkpoint,
        "epoch": ckpt.get("epoch", None),
        "device": str(device),
        "num_val_episodes": int(results["Q_pred"].shape[0]),
        "P_frob_rel": metrics.get("stats_p", {}),
        "Q_frob_rel": metrics["stats_q"],
        "R_frob_rel": metrics["stats_r"],
        "R2_P": metrics.get("stats_r2_p", {}),
        "R2_Q": metrics["stats_r2_q"],
        "R2_R": metrics["stats_r2_r"],
        "riccati_residual": metrics["stats_ric"],
        "riccati_residual_after_mint": metrics.get("stats_ric_after_mint", {}),
        "P_frob_rel_after_mint": metrics.get("stats_p_after_mint", {}),
        "Q_frob_rel_after_mint": metrics.get("stats_q_after_mint", {}),
        "R_frob_rel_after_mint": metrics.get("stats_r_after_mint", {}),
        "R2_P_after_mint": metrics.get("stats_r2_p_after_mint", {}),
        "R2_Q_after_mint": metrics.get("stats_r2_q_after_mint", {}),
        "R2_R_after_mint": metrics.get("stats_r2_r_after_mint", {}),
    }
    with open(os.path.join(args.out_dir, "eval_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[done] results saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
