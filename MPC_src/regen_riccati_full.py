"""Riccati coherence histogram — full 461 val samples, same style as original EN version."""
import os, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_eval import (HybridControllerModel, MPCDataset,
                      log_diag_to_matrix, denormalize_log,
                      mint_projection, riccati_residual)
from torch.utils.data import DataLoader, Subset

RUN = Path("mpc_pqr_run_logspace_norm")
OUT = RUN / "report_pdf_ro"
OUT.mkdir(exist_ok=True)
ckpt = torch.load(RUN / "best.pt", map_location="cpu", weights_only=False)
cfg = ckpt["args"]
stats_in = ckpt.get("input_stats", {})
stats_log = ckpt.get("stats", {})

d_seq = int(cfg["d_seq"]); d_static = int(cfg["d_static"])
n = int(cfg["n"]); m = int(cfg["m"])

model = HybridControllerModel(
    d_seq=d_seq, d_static=d_static,
    d_model=int(cfg["d_model"]), n=n, m=m,
    tx_layers=int(cfg["tx_layers"]), tx_heads=int(cfg["tx_heads"]),
    dropout=float(cfg["dropout"]))
model.load_state_dict(ckpt["model_state"]); model.eval()

mean_P = torch.tensor(stats_log["mean_P"], dtype=torch.float32)
std_P  = torch.tensor(stats_log["std_P"],  dtype=torch.float32)
mean_Q = torch.tensor(stats_log["mean_Q"], dtype=torch.float32)
std_Q  = torch.tensor(stats_log["std_Q"],  dtype=torch.float32)
mean_R = torch.tensor(stats_log["mean_R"], dtype=torch.float32)
std_R  = torch.tensor(stats_log["std_R"],  dtype=torch.float32)

ds = MPCDataset(root_dir="mpc_pqr_dataset_streaming_debug_500_samples",
                drop_duplicate_seq_feature=(d_seq == 4))
ds.set_normalization_stats(stats_in.get("seq_mean"), stats_in.get("seq_std"),
                            stats_in.get("static_mean"), stats_in.get("static_std"))

# Same val split as run_eval.py
rng = np.random.RandomState(42)
n_total = len(ds); n_val = int(float(cfg["val_frac"]) * n_total)
indices = list(range(n_total)); rng.shuffle(indices)
val_idx = indices[:n_val]
print(f"[data] {len(val_idx)} val samples")

val_ds = Subset(ds, val_idx)
loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

all_P_pred, all_Q_pred, all_R_pred, all_A, all_B = [], [], [], [], []
with torch.no_grad():
    for batch in loader:
        ts = batch["tokens_seq"]; tst = batch["tokens_static"]
        p_log_n, q_log_n, r_log_n, _ = model(ts, tst)
        p_log = denormalize_log(p_log_n, mean_P, std_P)
        q_log = denormalize_log(q_log_n, mean_Q, std_Q)
        r_log = denormalize_log(r_log_n, mean_R, std_R)
        all_P_pred.append(log_diag_to_matrix(p_log).cpu())
        all_Q_pred.append(log_diag_to_matrix(q_log).cpu())
        all_R_pred.append(log_diag_to_matrix(r_log).cpu())
        all_A.append(batch["A"]); all_B.append(batch["B"])

P_pred = torch.cat(all_P_pred); Q_pred = torch.cat(all_Q_pred); R_pred = torch.cat(all_R_pred)
A = torch.cat(all_A); B = torch.cat(all_B)

print("[mint] running projection on full val set...")
P_proj, Q_proj, R_proj = mint_projection(P_pred, Q_pred, R_pred, A, B, n_iter=3)

ric_before = riccati_residual(P_pred, Q_pred, R_pred, A, B).numpy()
ric_after = riccati_residual(P_proj, Q_proj, R_proj, A, B).numpy()
print(f"[stats] before: median={np.median(ric_before):.2f}, mean={ric_before.mean():.2f}")
print(f"[stats] after:  median={np.median(ric_after):.2f},  mean={ric_after.mean():.2f}")

# ============ Wide format for poster column ============
fig, ax = plt.subplots(figsize=(11, 4))
ax.hist(ric_before, bins=40, alpha=0.6, label="înainte de MinT", color="steelblue")
ax.hist(ric_after,  bins=40, alpha=0.6, label="după MinT",       color="coral")
ax.set_xlabel("Reziduul Riccati (norma Frobenius)", fontsize=11)
ax.set_ylabel("Număr de scenarii", fontsize=11)
ax.set_title("Coerență fizică — înainte vs. după MinT", fontsize=13, fontweight="bold")
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_yscale("log")
ax.tick_params(axis="both", labelsize=10)

fig.tight_layout()
fig.savefig(OUT / "riccati_coherence.pdf", bbox_inches="tight")
fig.savefig(OUT / "riccati_coherence.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"[ok] {OUT / 'riccati_coherence.pdf'}")
