"""
Standalone evaluation for the log10-diagonal P/Q/R Transformer model.
Reconstructs architecture from checkpoint, runs R^2 + Riccati + MinT.
"""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# Model architecture (must match training)
# ============================================================

class TransformerStack(nn.Module):
    def __init__(self, d_model=256, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.out_norm(self.enc(x))


class StaticEncoder(nn.Module):
    def __init__(self, d_static, d_model=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_static),
            nn.Linear(d_static, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


class DiagHead(nn.Module):
    def __init__(self, d_model, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class PQRHead(nn.Module):
    def __init__(self, d_model, n, m):
        super().__init__()
        self.p_head = DiagHead(d_model, n)
        self.q_head = DiagHead(d_model, n)
        self.r_head = DiagHead(d_model, m)

    def forward(self, h):
        return self.p_head(h), self.q_head(h), self.r_head(h)


class HybridControllerModel(nn.Module):
    def __init__(self, d_seq, d_static, d_model=256, n=2, m=1,
                 tx_layers=2, tx_heads=4, dropout=0.1):
        super().__init__()
        self.n = n
        self.m = m

        self.seq_proj = nn.Sequential(
            nn.Linear(d_seq, d_model),
            nn.LayerNorm(d_model),
        )
        self.pre_tf = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.transformer = TransformerStack(d_model, tx_layers, tx_heads, dropout)

        self.static_encoder = StaticEncoder(d_static, d_model)

        self.global_fusion = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.policy_fusion = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
        )

        self.pqr_head = PQRHead(d_model, n, m)
        self.policy_head = DiagHead(d_model, m)

    def forward(self, tokens_seq, tokens_static):
        h = self.seq_proj(tokens_seq)       # (B,T,d_model)
        h = self.pre_tf(h)
        h = self.transformer(h)             # (B,T,d_model)

        h_seq_global = h.mean(dim=1)        # (B,d_model)
        h_static = self.static_encoder(tokens_static)  # (B,d_model)

        h_global = self.global_fusion(torch.cat([h_seq_global, h_static], dim=-1))

        # Policy: per-timestep, fuse static into each timestep
        h_static_exp = h_static.unsqueeze(1).expand(-1, h.shape[1], -1)
        h_policy_seq = self.policy_fusion(torch.cat([h, h_static_exp], dim=-1))
        u_hat = self.policy_head(h_policy_seq)  # (B,T,m)

        # P, Q, R log10-diagonal
        p_log, q_log, r_log = self.pqr_head(h_global)

        return p_log, q_log, r_log, u_hat


# ============================================================
# Dataset (matches MPCDataset)
# ============================================================

from torch.utils.data import Dataset


class MPCDataset(Dataset):
    def __init__(self, root_dir, drop_duplicate_seq_feature=False,
                 duplicate_seq_feature_idx=3):
        self.root_dir = Path(root_dir)
        self.samples_dir = self.root_dir / "samples"
        self.drop_duplicate = drop_duplicate_seq_feature
        self.duplicate_idx = duplicate_seq_feature_idx
        self.seq_mean = self.seq_std = None
        self.static_mean = self.static_std = None

        self.sample_dirs = []
        for sd in sorted(self.samples_dir.glob("sample_*")):
            if not (sd / "meta.json").exists() or not (sd / "closed_loop.npz").exists():
                continue
            meta = json.load(open(sd / "meta.json"))
            if not meta.get("success", False):
                continue
            self.sample_dirs.append(sd)

    def set_normalization_stats(self, seq_mean, seq_std, static_mean, static_std):
        self.seq_mean = np.asarray(seq_mean, dtype=np.float32) if seq_mean is not None else None
        self.seq_std = np.asarray(seq_std, dtype=np.float32) if seq_std is not None else None
        self.static_mean = np.asarray(static_mean, dtype=np.float32) if static_mean is not None else None
        self.static_std = np.asarray(static_std, dtype=np.float32) if static_std is not None else None

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sd = self.sample_dirs[idx]
        meta = json.load(open(sd / "meta.json"))

        sig = np.load(sd / "signals.npz")
        tokens_seq = np.array(sig["tokens_seq"], dtype=np.float32)
        tokens_static = np.array(sig["tokens_static"], dtype=np.float32)

        cl = np.load(sd / "closed_loop.npz")
        u = np.array(cl["u_traj"], dtype=np.float32)
        if u.ndim == 1:
            u = u[:, None]

        tgt = meta["target"]
        P0 = np.array(tgt["P_raw"], dtype=np.float32)
        Q0 = np.array(tgt["Q_raw"], dtype=np.float32)
        R0 = np.array(tgt["R_raw"], dtype=np.float32)
        A_d = np.array(meta["model"]["A_d"], dtype=np.float32)
        B_d = np.array(meta["model"]["B_d"], dtype=np.float32)

        if self.drop_duplicate:
            keep = [j for j in range(tokens_seq.shape[1]) if j != self.duplicate_idx]
            tokens_seq = tokens_seq[:, keep]

        if self.seq_mean is not None:
            tokens_seq = (tokens_seq - self.seq_mean[None, :]) / self.seq_std[None, :]
        if self.static_mean is not None:
            tokens_static = (tokens_static - self.static_mean) / self.static_std

        return {
            "tokens_seq": torch.tensor(tokens_seq),
            "tokens_static": torch.tensor(tokens_static),
            "P0": torch.tensor(P0), "Q0": torch.tensor(Q0), "R0": torch.tensor(R0),
            "A": torch.tensor(A_d), "B": torch.tensor(B_d),
            "u": torch.tensor(u),
        }


# ============================================================
# Metrics: R^2, Riccati, MinT
# ============================================================

def frob_rel(a, b, eps=1e-8):
    num = torch.linalg.norm(a - b, ord="fro", dim=(-2, -1))
    den = torch.linalg.norm(b, ord="fro", dim=(-2, -1)) + eps
    return num / den


def r2_score_matrix(pred, target):
    pf = pred.reshape(pred.shape[0], -1)
    tf = target.reshape(target.shape[0], -1)
    ss_res = ((tf - pf) ** 2).sum(dim=-1)
    ss_tot = ((tf - tf.mean(dim=-1, keepdim=True)) ** 2).sum(dim=-1)
    return 1.0 - ss_res / (ss_tot + 1e-8)


def riccati_residual(P, Q, R, A, B, eps=1e-8):
    AtPA = A.transpose(-1, -2) @ P @ A
    BtPA = B.transpose(-1, -2) @ P @ A
    BtPB = B.transpose(-1, -2) @ P @ B
    M = R + BtPB + eps * torch.eye(R.shape[-1], device=R.device).expand_as(R)
    MinvBtPA = torch.linalg.solve(M, BtPA)
    res = P - Q - AtPA + BtPA.transpose(-1, -2) @ MinvBtPA
    return torch.linalg.norm(res, ord="fro", dim=(-2, -1))


def dare_residual_np(P, Q, R, A, B, eps=1e-8):
    AtPA = A.T @ P @ A
    BtPA = B.T @ P @ A
    BtPB = B.T @ P @ B
    M = R + BtPB + eps * np.eye(R.shape[0])
    MinvBtPA = np.linalg.solve(M, BtPA)
    return P - Q - AtPA + BtPA.T @ MinvBtPA


def numerical_jacobian(P, Q, R, A, B, eps_fd=1e-5):
    n = P.shape[0]; m = R.shape[0]
    n2 = n * n; m2 = m * m
    dim_theta = 2 * n2 + m2
    theta = np.concatenate([P.ravel(), Q.ravel(), R.ravel()])
    c0 = dare_residual_np(P, Q, R, A, B).ravel()
    C = np.zeros((n2, dim_theta))
    for i in range(dim_theta):
        tp = theta.copy()
        tp[i] += eps_fd
        Pi = tp[:n2].reshape(n, n)
        Qi = tp[n2:2 * n2].reshape(n, n)
        Ri = tp[2 * n2:].reshape(m, m)
        ci = dare_residual_np(Pi, Qi, Ri, A, B).ravel()
        C[:, i] = (ci - c0) / eps_fd
    return C


def project_psd(M, eps=1e-6):
    Ms = 0.5 * (M + M.T)
    w, v = np.linalg.eigh(Ms)
    w = np.maximum(w, eps)
    return (v * w) @ v.T


def mint_projection(P_pred, Q_pred, R_pred, A, B, n_iter=3):
    A_np = A.cpu().numpy().astype(np.float64)
    B_np = B.cpu().numpy().astype(np.float64)
    P_np = P_pred.cpu().numpy().astype(np.float64)
    Q_np = Q_pred.cpu().numpy().astype(np.float64)
    R_np = R_pred.cpu().numpy().astype(np.float64)

    batch = P_pred.shape[0]
    n = P_pred.shape[1]; m = R_pred.shape[1]
    n2 = n * n

    P_out = np.copy(P_np); Q_out = np.copy(Q_np); R_out = np.copy(R_np)

    for b in range(batch):
        Ab, Bb = A_np[b], B_np[b]
        Pb, Qb, Rb = P_out[b], Q_out[b], R_out[b]
        for _ in range(n_iter):
            c_vec = dare_residual_np(Pb, Qb, Rb, Ab, Bb).ravel()
            if np.linalg.norm(c_vec) < 1e-10:
                break
            C = numerical_jacobian(Pb, Qb, Rb, Ab, Bb)
            theta = np.concatenate([Pb.ravel(), Qb.ravel(), Rb.ravel()])
            try:
                CCtinv_c = np.linalg.solve(C @ C.T + 1e-8 * np.eye(n2), c_vec)
                correction = C.T @ CCtinv_c
                theta_new = theta - correction
                Pb = project_psd(theta_new[:n2].reshape(n, n))
                Qb = project_psd(theta_new[n2:2 * n2].reshape(n, n))
                Rb = project_psd(theta_new[2 * n2:].reshape(m, m))
            except np.linalg.LinAlgError:
                break
        P_out[b] = Pb; Q_out[b] = Qb; R_out[b] = Rb

    return (torch.from_numpy(P_out).float().to(P_pred.device),
            torch.from_numpy(Q_out).float().to(Q_pred.device),
            torch.from_numpy(R_out).float().to(R_pred.device))


def stats(t):
    t_np = t.detach().cpu().numpy()
    t_np = t_np[~np.isnan(t_np)]
    if len(t_np) == 0:
        return {}
    pcts = np.percentile(t_np, [25, 50, 75, 95])
    return {
        "mean": float(t_np.mean()), "std": float(t_np.std()),
        "p25": float(pcts[0]), "median": float(pcts[1]),
        "p75": float(pcts[2]), "p95": float(pcts[3]),
    }


# ============================================================
# Main
# ============================================================

def log_diag_to_matrix(log_diag):
    """Convert (B, n) log10 predictions to (B, n, n) diagonal PSD matrices."""
    diag = torch.pow(10.0, log_diag)
    return torch.diag_embed(diag)


def denormalize_log(log_norm, mean, std):
    """log_norm = (log - mean)/std  →  log = log_norm * std + mean"""
    return log_norm * std.unsqueeze(0) + mean.unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device(args.device)
    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(args.checkpoint), "eval")
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["args"]
    print(f"[ckpt] epoch={ckpt.get('epoch')}  val_loss={ckpt.get('val_metrics', {}).get('loss', '?')}")

    input_stats = ckpt.get("input_stats", {})
    log_stats = ckpt.get("stats", {})

    seq_mean = torch.tensor(input_stats.get("seq_mean"), dtype=torch.float32)
    seq_std = torch.tensor(input_stats.get("seq_std"), dtype=torch.float32)
    static_mean = torch.tensor(input_stats.get("static_mean"), dtype=torch.float32)
    static_std = torch.tensor(input_stats.get("static_std"), dtype=torch.float32)

    mean_P = torch.tensor(log_stats["mean_P"], dtype=torch.float32, device=device)
    std_P = torch.tensor(log_stats["std_P"], dtype=torch.float32, device=device)
    mean_Q = torch.tensor(log_stats["mean_Q"], dtype=torch.float32, device=device)
    std_Q = torch.tensor(log_stats["std_Q"], dtype=torch.float32, device=device)
    mean_R = torch.tensor(log_stats["mean_R"], dtype=torch.float32, device=device)
    std_R = torch.tensor(log_stats["std_R"], dtype=torch.float32, device=device)

    d_seq = int(cfg["d_seq"])
    d_static = int(cfg["d_static"])
    n = int(cfg["n"]); m = int(cfg["m"])

    model = HybridControllerModel(
        d_seq=d_seq, d_static=d_static,
        d_model=int(cfg["d_model"]),
        n=n, m=m,
        tx_layers=int(cfg["tx_layers"]),
        tx_heads=int(cfg["tx_heads"]),
        dropout=float(cfg["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] loaded, {n_params/1e6:.2f}M params")

    # Dataset (apply same normalization + drop duplicate feature if d_seq==4)
    ds = MPCDataset(
        root_dir=args.data_root,
        drop_duplicate_seq_feature=(d_seq == 4),  # raw is 5 → drop to 4
        duplicate_seq_feature_idx=3,
    )
    ds.set_normalization_stats(
        input_stats.get("seq_mean"), input_stats.get("seq_std"),
        input_stats.get("static_mean"), input_stats.get("static_std"),
    )

    # Same train/val split
    rng = np.random.RandomState(args.seed)
    n_total = len(ds)
    n_val = int(float(cfg["val_frac"]) * n_total)
    indices = list(range(n_total))
    rng.shuffle(indices)
    val_idx = indices[:n_val]

    from torch.utils.data import Subset
    val_ds = Subset(ds, val_idx)
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[data] total={n_total}  val={len(val_ds)}")

    # Inference
    all_P_pred, all_P0 = [], []
    all_Q_pred, all_Q0 = [], []
    all_R_pred, all_R0 = [], []
    all_A, all_B = [], []
    all_u_pred, all_u = [], []

    with torch.no_grad():
        for batch in loader:
            ts = batch["tokens_seq"].to(device)
            tst = batch["tokens_static"].to(device)
            p_log_norm, q_log_norm, r_log_norm, u_hat = model(ts, tst)

            # Denormalize
            p_log = denormalize_log(p_log_norm, mean_P, std_P)
            q_log = denormalize_log(q_log_norm, mean_Q, std_Q)
            r_log = denormalize_log(r_log_norm, mean_R, std_R)

            # Convert log10 → diagonal matrices
            P_pred = log_diag_to_matrix(p_log)
            Q_pred = log_diag_to_matrix(q_log)
            R_pred = log_diag_to_matrix(r_log)

            all_P_pred.append(P_pred.cpu()); all_P0.append(batch["P0"])
            all_Q_pred.append(Q_pred.cpu()); all_Q0.append(batch["Q0"])
            all_R_pred.append(R_pred.cpu()); all_R0.append(batch["R0"])
            all_A.append(batch["A"]); all_B.append(batch["B"])
            all_u_pred.append(u_hat.cpu()); all_u.append(batch["u"])

    P_pred = torch.cat(all_P_pred); P0 = torch.cat(all_P0)
    Q_pred = torch.cat(all_Q_pred); Q0 = torch.cat(all_Q0)
    R_pred = torch.cat(all_R_pred); R0 = torch.cat(all_R0)
    A = torch.cat(all_A); B = torch.cat(all_B)
    u_pred = torch.cat(all_u_pred); u_gt = torch.cat(all_u)

    print(f"[eval] predictions: P{tuple(P_pred.shape)}  Q{tuple(Q_pred.shape)}  R{tuple(R_pred.shape)}")

    # ── Metrics BEFORE MinT ──
    p_err = frob_rel(P_pred, P0)
    q_err = frob_rel(Q_pred, Q0)
    r_err = frob_rel(R_pred, R0)
    r2_p = r2_score_matrix(P_pred, P0)
    r2_q = r2_score_matrix(Q_pred, Q0)
    r2_r = r2_score_matrix(R_pred, R0)
    ric = riccati_residual(P_pred, Q_pred, R_pred, A, B)

    # ── MinT projection ──
    print("[eval] running MinT projection...")
    P_proj, Q_proj, R_proj = mint_projection(P_pred, Q_pred, R_pred, A, B, n_iter=3)

    p_err_a = frob_rel(P_proj, P0)
    q_err_a = frob_rel(Q_proj, Q0)
    r_err_a = frob_rel(R_proj, R0)
    r2_p_a = r2_score_matrix(P_proj, P0)
    r2_q_a = r2_score_matrix(Q_proj, Q0)
    r2_r_a = r2_score_matrix(R_proj, R0)
    ric_a = riccati_residual(P_proj, Q_proj, R_proj, A, B)

    # ── Print ──
    print("\n" + "=" * 70)
    print(f"EVALUATION  —  {len(P_pred)} val samples  —  checkpoint epoch {ckpt.get('epoch')}")
    print("=" * 70)

    def pline(name, s):
        if s:
            print(f"  {name:22s}  mean={s['mean']:.4f}  median={s['median']:.4f}  p95={s['p95']:.4f}")

    print("\n-- Frobenius Relative Error (before MinT) --")
    pline("P", stats(p_err)); pline("Q", stats(q_err)); pline("R", stats(r_err))

    print("\n-- R² Score (before MinT) --")
    pline("R²(P)", stats(r2_p)); pline("R²(Q)", stats(r2_q)); pline("R²(R)", stats(r2_r))

    print("\n-- Riccati Coherence --")
    pline("Residual before", stats(ric))
    pline("Residual after MinT", stats(ric_a))

    print("\n-- Frobenius Relative Error (after MinT) --")
    pline("P", stats(p_err_a)); pline("Q", stats(q_err_a)); pline("R", stats(r_err_a))

    print("\n-- R² Score (after MinT) --")
    pline("R²(P)", stats(r2_p_a)); pline("R²(Q)", stats(r2_q_a)); pline("R²(R)", stats(r2_r_a))

    print("=" * 70)

    # ── Save ──
    summary = {
        "checkpoint": args.checkpoint,
        "epoch": ckpt.get("epoch"),
        "num_val_samples": len(P_pred),
        "before_mint": {
            "FRE_P": stats(p_err), "FRE_Q": stats(q_err), "FRE_R": stats(r_err),
            "R2_P": stats(r2_p), "R2_Q": stats(r2_q), "R2_R": stats(r2_r),
            "riccati_residual": stats(ric),
        },
        "after_mint": {
            "FRE_P": stats(p_err_a), "FRE_Q": stats(q_err_a), "FRE_R": stats(r_err_a),
            "R2_P": stats(r2_p_a), "R2_Q": stats(r2_q_a), "R2_R": stats(r2_r_a),
            "riccati_residual": stats(ric_a),
        },
    }
    with open(os.path.join(args.out_dir, "eval_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── Plots ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, data, title in zip(
        axes[0],
        [p_err.numpy(), q_err.numpy(), r_err.numpy()],
        ["FRE(P) before", "FRE(Q) before", "FRE(R) before"],
    ):
        data = data[~np.isnan(data)]
        ax.hist(data, bins=40, color="steelblue", alpha=0.8)
        ax.axvline(np.median(data), color="red", linestyle="--",
                   label=f"med={np.median(data):.3f}")
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    for ax, data, title in zip(
        axes[1],
        [p_err_a.numpy(), q_err_a.numpy(), r_err_a.numpy()],
        ["FRE(P) after MinT", "FRE(Q) after MinT", "FRE(R) after MinT"],
    ):
        data = data[~np.isnan(data)]
        ax.hist(data, bins=40, color="coral", alpha=0.8)
        ax.axvline(np.median(data), color="red", linestyle="--",
                   label=f"med={np.median(data):.3f}")
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fre_distributions.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(ric.numpy(), bins=40, alpha=0.6, label="before MinT", color="steelblue")
    ax.hist(ric_a.numpy(), bins=40, alpha=0.6, label="after MinT", color="coral")
    ax.set_xlabel("Riccati residual (Frobenius)"); ax.set_ylabel("Count")
    ax.set_title("Riccati coherence: before vs after MinT projection")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "riccati_coherence.png"), dpi=120)
    plt.close(fig)

    print(f"\n[done] saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
