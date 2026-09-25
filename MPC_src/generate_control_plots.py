"""
generate_control_plots.py
=========================
Generates control-oriented evaluation figures and stability analysis
for the best Transformer-driven MPC cost matrix adaptation model.

Produces:
  cl_eigenvalues_unit_circle.png  -- poles of closed-loop system (stability)
  cl_trajectory_examples.png      -- angular velocity tracking + control effort
  cl_metrics_boxplot.png          -- IAE / settling time / overshoot boxplots

Run from MPC_new-2/ root:
  python MPC_src/generate_control_plots.py

Or with explicit paths:
  python MPC_src/generate_control_plots.py --checkpoint best/run/best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import MPCDataset, get_manifest
from model import HybridControllerModel
from train import split_indices

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    print("[warning] cvxpy not installed — closed-loop simulation skipped; only eigenvalue plots")


# ── helpers ───────────────────────────────────────────────────

def _load_normalization(run_dir: Path):
    data = np.load(run_dir / "normalization.npz")
    return data["seq_mean"], data["seq_std"], data["static_mean"], data["static_std"]


def _load_config(checkpoint: Path) -> dict:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config: dict = {}
    cfg_path = checkpoint.parent / "config.json"
    if cfg_path.exists():
        with cfg_path.open() as f:
            config = json.load(f)
    config.update(ckpt.get("args", {}))
    return config


@torch.no_grad()
def _predict_pqr_batch(model, batch, device, n, m):
    """Return (P_diag, Q_diag, R_diag) as (B,n), (B,n), (B,m) numpy arrays."""
    model.eval()
    ts = batch["tokens_seq"].to(device)
    tst = batch["tokens_static"].to(device)
    pred = model(ts, tst)

    if len(pred) == 4:
        Pp, Qp, Rp, _ = pred
    else:
        Pp, Qp, Rp = pred

    pred_log = torch.cat([Pp, Qp, Rp], dim=-1)
    P_diag = torch.pow(10.0, pred_log[:, :n]).cpu().numpy()
    Q_diag = torch.pow(10.0, pred_log[:, n:n + n]).cpu().numpy()
    R_diag = torch.pow(10.0, pred_log[:, n + n:n + n + m]).cpu().numpy()
    return P_diag, Q_diag, R_diag


# ── stability analysis ────────────────────────────────────────

def _cl_eigenvalues(A, B, P_diag, Q_diag, R_diag):
    """Closed-loop eigenvalues via LQR gain K = (R + B'PB)^{-1} B'PA."""
    P = np.diag(P_diag.clip(1e-12))
    R = np.diag(R_diag.clip(1e-12))
    try:
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        return np.linalg.eigvals(A - B @ K)
    except np.linalg.LinAlgError:
        return np.array([])


def _pick_one_per_type(ds, val_idx):
    """Pick one sample per reference type (sine, step, multi_step) plus one extra step."""
    by_type = {}
    extras = []
    for si in val_idx:
        meta = json.load(open(ds.sample_dirs[si] / "meta.json"))
        rt = meta["scenario"].get("reference_type", "unknown")
        if rt not in by_type:
            by_type[rt] = si
        elif rt == "step" and "step2" not in by_type:
            by_type["step2"] = si
        else:
            extras.append(si)
        if len(by_type) >= 4:
            break
    chosen = list(by_type.values())[:4]
    for si in extras:
        if si not in chosen:
            chosen.append(si)
        if len(chosen) == 4:
            break
    return chosen


_TYPE_LABELS = {
    "sine": "Sinusoidal reference",
    "step": "Step reference",
    "step2": "Step reference",
    "multi_step": "Multi-step reference",
}

_PALETTE = {
    "target": "#2ca02c",
    "tfmr":   "#1f77b4",
    "ref":    "#222222",
}


def plot_target_eigenvalues_4scenarios(ds, val_idx, out_dir):
    """2x2 grid: one sample per ref_type, offline-optimized poles — publication style."""
    chosen = _pick_one_per_type(ds, val_idx)
    theta = np.linspace(0, 2 * np.pi, 400)

    fig, axes = plt.subplots(2, 2, figsize=(8, 7.5))
    axes = axes.flatten()

    for ax, si in zip(axes, chosen):
        meta = json.load(open(ds.sample_dirs[si] / "meta.json"))
        A  = np.array(meta["model"]["A_d"])
        B  = np.array(meta["model"]["B_d"])
        p  = np.array(meta["target"]["P_diag"])
        q  = np.array(meta["target"]["Q_diag"])
        r  = np.array(meta["target"]["R_diag"])
        rt = meta["scenario"].get("reference_type", "unknown")
        Ts = float(meta["model"]["Ts"])

        eigs = np.array(_cl_eigenvalues(A, B, p, q, r))

        # unit disc shading
        ax.fill(np.cos(theta), np.sin(theta), color="#f0f0f0", zorder=0)
        ax.plot(np.cos(theta), np.sin(theta), color="#555555",
                linewidth=1.2, zorder=1)
        ax.axhline(0, color="#bbbbbb", linewidth=0.6, zorder=1)
        ax.axvline(0, color="#bbbbbb", linewidth=0.6, zorder=1)

        # poles
        ax.scatter(np.real(eigs), np.imag(eigs),
                   s=140, color=_PALETTE["target"], zorder=5,
                   edgecolors="white", linewidths=1.2)

        for e in eigs:
            offset = (8, 5) if np.real(e) > 0.1 else (8, -14)
            ax.annotate(f"$|\\lambda|={abs(e):.3f}$",
                        xy=(np.real(e), np.imag(e)),
                        xytext=offset, textcoords="offset points",
                        fontsize=8, color="#1a5c1a",
                        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.6))

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal")
        label = _TYPE_LABELS.get(rt, rt)
        ax.set_title(f"{label}", fontsize=10, fontweight="bold")
        ax.set_xlabel(r"Re($\lambda$)", fontsize=9)
        ax.set_ylabel(r"Im($\lambda$)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])

    fig.suptitle(
        "Closed-Loop Pole Locations — Offline-Optimized $P$, $Q$, $R$",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = out_dir / "cl_eigenvalues_target_4scenarios.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[eig] saved {out}")


def plot_transformer_eigenvalues(model, ds, val_idx, device, n, m, out_dir, max_eig=300):
    """Single plot: all validation samples, Transformer-adapted P/Q/R poles."""
    loader = DataLoader(
        torch.utils.data.Subset(ds, val_idx[:max_eig]),
        batch_size=64, shuffle=False, num_workers=0,
    )
    eigs_pred, eigs_tgt = [], []

    for batch in loader:
        P_pred, Q_pred, R_pred = _predict_pqr_batch(model, batch, device, n, m)
        A_b = batch["A"].numpy()
        B_b = batch["B"].numpy()
        P_t = torch.diagonal(batch["P0"], dim1=-2, dim2=-1).numpy()
        Q_t = torch.diagonal(batch["Q0"], dim1=-2, dim2=-1).numpy()
        R_t = torch.diagonal(batch["R0"], dim1=-2, dim2=-1).numpy()

        for i in range(len(A_b)):
            A, B = A_b[i], B_b[i]
            eigs_pred.extend(_cl_eigenvalues(A, B, P_pred[i], Q_pred[i], R_pred[i]).tolist())
            eigs_tgt.extend(_cl_eigenvalues(A, B, P_t[i], Q_t[i], R_t[i]).tolist())

    eigs_pred = np.array(eigs_pred)
    eigs_tgt  = np.array(eigs_tgt)
    pct_pred = np.mean(np.abs(eigs_pred) <= 1.0 + 1e-6) * 100
    pct_tgt  = np.mean(np.abs(eigs_tgt)  <= 1.0 + 1e-6) * 100

    theta = np.linspace(0, 2 * np.pi, 400)
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.fill_between(np.cos(theta), np.sin(theta), alpha=0.07, color="gray")
    ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=1.5, alpha=0.7)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.4)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.4)
    ax.scatter(np.real(eigs_pred), np.imag(eigs_pred), s=18, color="tab:blue",
               alpha=0.45, label=f"Transformer  (|$\\lambda$|$\\leq$1: {pct_pred:.1f}%)")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.set_title("Transformer-Adapted P/Q/R", fontsize=12)
    ax.set_xlabel(r"Re($\lambda$)", fontsize=11)
    ax.set_ylabel(r"Im($\lambda$)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    fig.suptitle("Closed-Loop Eigenvalues — Transformer Predictions", fontsize=13)
    fig.tight_layout()
    out = out_dir / "cl_eigenvalues_transformer.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[eig] saved {out}")
    print(f"      Target stable: {pct_tgt:.1f}%   Transformer stable: {pct_pred:.1f}%")
    return pct_pred, pct_tgt


# ── MPC closed-loop simulation ────────────────────────────────

if HAS_CVXPY:
    class ParametricMPC:
        def __init__(self, A, B, C, E, u_min, u_max, du_min, du_max, horizon):
            n, m = A.shape[0], B.shape[1]
            self.x = cp.Variable((n, horizon + 1))
            self.u = cp.Variable((m, horizon))
            self.x0_p = cp.Parameter(n)
            self.u_prev_p = cp.Parameter(m)
            self.d_p = cp.Parameter(horizon)
            self.x_ref_p = cp.Parameter((n, horizon + 1))
            self.sqrt_p = cp.Parameter(n, nonneg=True)
            self.sqrt_q = cp.Parameter(n, nonneg=True)
            self.sqrt_r = cp.Parameter(m, nonneg=True)

            cost = 0.0
            cons = [self.x[:, 0] == self.x0_p]
            for k in range(horizon):
                cons += [
                    self.x[:, k + 1] == A @ self.x[:, k] + B @ self.u[:, k] + E.flatten() * self.d_p[k],
                    self.u[:, k] >= u_min, self.u[:, k] <= u_max,
                ]
                if k == 0:
                    cons += [self.u[:, k] - self.u_prev_p >= du_min,
                             self.u[:, k] - self.u_prev_p <= du_max]
                else:
                    cons += [self.u[:, k] - self.u[:, k - 1] >= du_min,
                             self.u[:, k] - self.u[:, k - 1] <= du_max]
                x_err = self.x[:, k] - self.x_ref_p[:, k]
                cost += cp.sum_squares(cp.multiply(self.sqrt_q, x_err))
                cost += cp.sum_squares(cp.multiply(self.sqrt_r, self.u[:, k]))
            x_err_N = self.x[:, horizon] - self.x_ref_p[:, horizon]
            cost += cp.sum_squares(cp.multiply(self.sqrt_p, x_err_N))
            self.problem = cp.Problem(cp.Minimize(cost), cons)

        def solve(self, x0, u_prev, x_ref_hor, d_hor, p_diag, q_diag, r_diag):
            self.x0_p.value = np.asarray(x0, dtype=float)
            self.u_prev_p.value = np.array([u_prev], dtype=float)
            self.x_ref_p.value = np.asarray(x_ref_hor, dtype=float)
            self.d_p.value = np.asarray(d_hor, dtype=float)
            self.sqrt_p.value = np.sqrt(np.maximum(p_diag, 1e-12))
            self.sqrt_q.value = np.sqrt(np.maximum(q_diag, 1e-12))
            self.sqrt_r.value = np.sqrt(np.maximum(r_diag, 1e-12))
            try:
                self.problem.solve(solver=cp.OSQP, warm_start=True, verbose=False, max_iter=5000)
            except Exception:
                return None, "exception"
            if self.problem.status not in ("optimal", "optimal_inaccurate"):
                return None, self.problem.status
            return float(self.u[:, 0].value.item()), self.problem.status

    def simulate_cl(mpc, A, B, C, E, p_diag, q_diag, r_diag,
                    x_ref_full, reference, disturbance, x0,
                    sim_steps, horizon, noise_std, seed):
        rng = np.random.default_rng(seed)
        n = A.shape[0]
        x = np.zeros((sim_steps + 1, n))
        y = np.zeros(sim_steps)
        u = np.zeros(sim_steps)
        x[0] = x0
        u_prev = 0.0

        for k in range(sim_steps):
            y[k] = float((C @ x[k]).item())
            x_meas = x[k].copy()
            x_meas[1] = y[k] + rng.normal(0.0, noise_std)
            d_hor = np.array([disturbance[min(k + j, sim_steps - 1)] for j in range(horizon)])
            x_ref_hor = np.column_stack(
                [x_ref_full[:, min(k + j, sim_steps - 1)] for j in range(horizon + 1)]
            )
            uk, status = mpc.solve(x_meas, u_prev, x_ref_hor, d_hor, p_diag, q_diag, r_diag)
            if uk is None:
                return {"feasible": False, "y": y[:k], "u": u[:k]}
            u[k] = uk
            x[k + 1] = A @ x[k] + B.flatten() * u[k] + E.flatten() * disturbance[k]
            u_prev = u[k]
            if np.any(np.abs(x[k + 1]) > 1e5):
                return {"feasible": False, "y": y[:k + 1], "u": u[:k + 1]}

        return {"feasible": True, "y": y, "u": u}


def _ctrl_metrics(y, u, reference, Ts, u_min, u_max):
    if len(y) == 0:
        return {}
    e = reference[:len(y)] - y
    ref_scale = max(1.0, float(np.max(np.abs(reference[:len(y)]))))
    iae_norm = float(np.sum(np.abs(e)) * Ts) / ref_scale
    ise_norm = float(np.sum(e ** 2) * Ts) / (ref_scale ** 2)
    rmse_norm = float(np.sqrt(np.mean(e ** 2))) / ref_scale
    final_err_norm = float(abs(e[-1])) / ref_scale
    sat_ratio = float(np.mean((np.abs(u - u_max) < 1e-3) | (np.abs(u - u_min) < 1e-3)))
    final_ref = float(reference[len(y) - 1])
    overshoot = float(max(0.0, (np.max(y) - final_ref) / abs(final_ref))) if abs(final_ref) > 1e-8 else 0.0
    settling = float(len(y) * Ts)
    band = 0.02 * max(1.0, abs(final_ref))
    for k in range(len(y)):
        if np.all(np.abs(y[k:] - final_ref) <= band):
            settling = float(k * Ts)
            break
    return {
        "iae_norm": iae_norm, "ise_norm": ise_norm, "rmse_norm": rmse_norm,
        "final_error_norm": final_err_norm, "sat_ratio": sat_ratio,
        "overshoot": overshoot, "settling_time": settling,
    }


def run_sim_and_plot(model, ds, sample_indices, device, n, m, out_dir, max_sim=30, seed=42):
    val_idx = sample_indices
    all_metrics = []
    traj_samples = []
    traj_candidates = []
    mpc_cache = {}

    for ii, si in enumerate(val_idx[:max_sim]):
        if (ii + 1) % 5 == 0:
            print(f"  [{ii + 1}/{min(max_sim, len(val_idx))}]")

        sample_dir = ds.sample_dirs[si]
        meta = json.load(open(sample_dir / "meta.json"))
        sig = np.load(sample_dir / "signals.npz")

        A_d = np.array(meta["model"]["A_d"])
        B_d = np.array(meta["model"]["B_d"])
        C_d = np.array(meta["model"]["C_d"])
        E_d = np.array(meta["model"]["E_d"])
        Ts = float(meta["model"]["Ts"])
        horizon = int(meta["constraints"]["horizon"])
        sim_steps = int(meta["constraints"]["sim_steps"])
        u_min = float(meta["constraints"]["u_min"])
        u_max = float(meta["constraints"]["u_max"])
        du_min = float(meta["constraints"]["du_min"])
        du_max = float(meta["constraints"]["du_max"])
        reference = np.array(sig["reference"])
        disturbance = np.array(sig["disturbance"])
        x_ref_full = np.array(sig["x_ref"])
        x0 = np.array(meta["scenario"]["x0"])
        noise_std = float(meta["scenario"].get("measurement_noise_std", 0.0))
        p_gt = np.array(meta["target"]["P_diag"])
        q_gt = np.array(meta["target"]["Q_diag"])
        r_gt = np.array(meta["target"]["R_diag"])

        item = ds[si]
        batch_s = {k: v.unsqueeze(0) for k, v in item.items() if isinstance(v, torch.Tensor)}
        p_pred, q_pred, r_pred = _predict_pqr_batch(model, batch_s, device, n, m)
        p_pred, q_pred, r_pred = p_pred[0], q_pred[0], r_pred[0]

        key = (tuple(A_d.flatten().round(8)), horizon)
        if key not in mpc_cache:
            mpc_cache[key] = ParametricMPC(A_d, B_d, C_d, E_d,
                                           u_min, u_max, du_min, du_max, horizon)
        mpc = mpc_cache[key]

        row = {"sample_idx": si}
        sims = {}
        for vname, (pd, qd, rd) in [("target", (p_gt, q_gt, r_gt)),
                                     ("tfmr", (p_pred, q_pred, r_pred))]:
            sim = simulate_cl(mpc, A_d, B_d, C_d, E_d, pd, qd, rd,
                              x_ref_full, reference, disturbance, x0,
                              sim_steps, horizon, noise_std, seed=200000 + si)
            sims[vname] = sim
            if sim["feasible"]:
                row[vname] = _ctrl_metrics(sim["y"], sim["u"], reference, Ts, u_min, u_max)
        all_metrics.append(row)

        tgt_ok = sims.get("target", {}).get("feasible", False)
        tfmr_ok = sims.get("tfmr", {}).get("feasible", False)
        ref_scale = float(np.max(np.abs(reference))) if len(reference) else 1.0
        tgt_sane = tgt_ok and float(np.max(np.abs(sims["target"]["y"]))) < 4.0 * max(ref_scale, 1.0)
        tfmr_sane = tfmr_ok and float(np.max(np.abs(sims["tfmr"]["y"]))) < 4.0 * max(ref_scale, 1.0)

        if tgt_ok and tfmr_ok and tgt_sane and tfmr_sane and "target" in row and "tfmr" in row:
            ref_scale = max(ref_scale, 1.0)
            jump_target = float(np.max(np.abs(np.diff(sims["target"]["y"])))) / ref_scale
            jump_tfmr = float(np.max(np.abs(np.diff(sims["tfmr"]["y"])))) / ref_scale
            score = (
                row["target"]["iae_norm"]
                + row["tfmr"]["iae_norm"]
                + 0.35 * (jump_target + jump_tfmr)
                + 0.25 * abs(row["target"]["final_error_norm"] - row["tfmr"]["final_error_norm"])
            )
            traj_candidates.append({
                "sample_idx": si, "Ts": Ts, "reference": reference,
                "ref_type": meta["scenario"].get("reference_type", ""),
                "difficulty": meta["scenario"].get("difficulty", ""),
                "u_max": u_max, "target": sims["target"], "tfmr": sims["tfmr"],
                "score": score,
            })

    if traj_candidates:
        by_type = {}
        for tr in sorted(traj_candidates, key=lambda item: item["score"]):
            if tr["ref_type"] not in by_type:
                by_type[tr["ref_type"]] = tr
        preferred = ["step", "sine", "multi_step"]
        traj_samples = [by_type[rt] for rt in preferred if rt in by_type]
        for tr in sorted(traj_candidates, key=lambda item: item["score"]):
            if tr not in traj_samples:
                traj_samples.append(tr)
            if len(traj_samples) >= 4:
                break
        print("[sim] selected trajectory samples:")
        for tr in traj_samples[:4]:
            print(f"      sample_idx={tr['sample_idx']} ref={tr['ref_type']} score={tr['score']:.3f}")

    # ── trajectory plot — 2×2, shared legend ──
    n_show = min(len(traj_samples), 4)
    if n_show > 0:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        axes_flat = axes.flatten()

        lines_handles = []
        for i, tr in enumerate(traj_samples[:n_show]):
            ax = axes_flat[i]
            Ts_i = tr["Ts"]
            ref  = tr["reference"]
            t    = np.arange(len(ref)) * Ts_i

            l0, = ax.plot(t, ref, color=_PALETTE["ref"], lw=1.6,
                          ls=(0, (5, 3)), alpha=0.85,
                          label=r"$\omega_{\mathrm{ref}}$")
            l1 = l2 = None
            if tr["target"]["feasible"]:
                y_o = tr["target"]["y"]
                l1, = ax.plot(np.arange(len(y_o)) * Ts_i, y_o,
                              color=_PALETTE["target"], lw=2.0,
                              label="Offline-optimized $P,Q,R$")
            if tr["tfmr"]["feasible"]:
                y_t = tr["tfmr"]["y"]
                l2, = ax.plot(np.arange(len(y_t)) * Ts_i, y_t,
                              color=_PALETTE["tfmr"], lw=2.0, ls="--",
                              label="Transformer (adapted $P,Q,R$)")

            label = _TYPE_LABELS.get(tr["ref_type"], tr["ref_type"])
            ax.set_title(label, fontsize=10, fontweight="bold")
            ax.set_xlabel("Time [s]", fontsize=9)
            ax.set_ylabel(r"$\omega\,$ [rad/s]", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.25, linestyle=":")
            ax.spines[["top", "right"]].set_visible(False)

            if i == 0:
                lines_handles = [h for h in [l0, l1, l2] if h is not None]

        # hide unused subplot if fewer than 4 samples
        for j in range(n_show, 4):
            axes_flat[j].set_visible(False)

        fig.legend(handles=lines_handles,
                   loc="lower center", ncol=3, fontsize=9,
                   frameon=True, framealpha=0.9,
                   bbox_to_anchor=(0.5, -0.03))
        fig.suptitle("Closed-Loop Angular Velocity Tracking",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        out = out_dir / "cl_trajectory_examples.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[sim] saved {out}")

        # Compact paper version: same information, less vertical space.
        fig, axes = plt.subplots(1, n_show, figsize=(11, 2.45), sharex=False)
        if n_show == 1:
            axes = [axes]

        lines_handles = []
        for i, tr in enumerate(traj_samples[:n_show]):
            ax = axes[i]
            Ts_i = tr["Ts"]
            ref = tr["reference"]
            t = np.arange(len(ref)) * Ts_i

            l0, = ax.plot(t, ref, color=_PALETTE["ref"], lw=1.25,
                          ls=(0, (5, 3)), alpha=0.85,
                          label=r"$\omega_{\mathrm{ref}}$")
            l1 = l2 = None
            if tr["target"]["feasible"]:
                y_o = tr["target"]["y"]
                l1, = ax.plot(np.arange(len(y_o)) * Ts_i, y_o,
                              color=_PALETTE["target"], lw=1.65,
                              label="Offline MPC")
            if tr["tfmr"]["feasible"]:
                y_t = tr["tfmr"]["y"]
                l2, = ax.plot(np.arange(len(y_t)) * Ts_i, y_t,
                              color=_PALETTE["tfmr"], lw=1.65, ls="--",
                              label="Transformer MPC")

            label = _TYPE_LABELS.get(tr["ref_type"], tr["ref_type"])
            ax.set_title(label, fontsize=8.5, fontweight="bold")
            ax.set_xlabel("Time [s]", fontsize=8)
            if i == 0:
                ax.set_ylabel(r"$\omega$ [rad/s]", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.22, linestyle=":")
            ax.spines[["top", "right"]].set_visible(False)

            if i == 0:
                lines_handles = [h for h in [l0, l1, l2] if h is not None]

        fig.legend(handles=lines_handles, loc="lower center", ncol=3,
                   fontsize=8, frameon=True, framealpha=0.9,
                   bbox_to_anchor=(0.5, -0.04))
        fig.tight_layout(rect=[0, 0.14, 1, 1])
        out = out_dir / "cl_trajectory_compact.png"
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"[sim] saved {out}")

    # ── boxplot ──
    mk_list = ["iae_norm", "ise_norm", "final_error_norm", "settling_time", "overshoot"]
    titles = ["IAE (norm.)", "ISE (norm.)", "Final Error (norm.)", "Settling Time [s]", "Overshoot"]
    fig, axes = plt.subplots(1, len(mk_list), figsize=(15, 4.5))
    for ax, mk, ttl in zip(axes, mk_list, titles):
        data, labels = [], []
        for vname, color, label in [("target", "tab:green", "Offline-optimized"), ("tfmr", "tab:blue", "Transformer")]:
            vals = [r[vname][mk] for r in all_metrics
                    if vname in r and mk in r[vname] and np.isfinite(r[vname][mk])]
            data.append(vals)
            labels.append(label)
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True)
        for patch, color in zip(bp["boxes"], ["tab:green", "tab:blue"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(ttl, fontsize=10)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Control Performance: Oracle vs Transformer-Adapted MPC", fontsize=12)
    fig.tight_layout()
    out = out_dir / "cl_metrics_boxplot.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[sim] saved {out}")

    # ── print summary table ──
    print("\n" + "=" * 68)
    print(f"{'Metric':25s} | {'Offline-opt (med)':>16s} | {'Transformer (med)':>16s}")
    print("-" * 68)
    for mk in mk_list:
        vals_o = [r["target"][mk] for r in all_metrics
                  if "target" in r and np.isfinite(r["target"].get(mk, float("nan")))]
        vals_t = [r["tfmr"][mk] for r in all_metrics
                  if "tfmr" in r and np.isfinite(r["tfmr"].get(mk, float("nan")))]
        mo = float(np.median(vals_o)) if vals_o else float("nan")
        mt = float(np.median(vals_t)) if vals_t else float("nan")
        print(f"{mk:25s} | {mo:>16.4f} | {mt:>16.4f}")
    print("=" * 68)

    return all_metrics


# ── main ──────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--max_eig", type=int, default=300)
    ap.add_argument("--max_sim", type=int, default=30)
    ap.add_argument("--val_indices", type=int, nargs="+", default=None,
                    help="Specific val_indices to simulate (skips max_sim search)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    here = Path(__file__).parent
    root = here.parent
    checkpoint = Path(args.checkpoint) if args.checkpoint else root / "runs" / "paper_model" / "best.pt"
    run_dir = checkpoint.parent
    config = _load_config(checkpoint)
    data_root = args.data_root or config["data_root"]
    out_dir = Path(args.out_dir) if args.out_dir else root / "runs" / "paper_model" / "eval_control"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    manifest = get_manifest(data_root, target_scale=config.get("target_scale", "none"))
    n = manifest["n"]
    m = manifest["m"]

    model = HybridControllerModel(
        d_in=manifest["d_seq"],
        d_static=manifest["d_static"],
        d_model=int(config["d_model"]),
        n=n, m=m,
        mamba_layers=int(config.get("mamba_layers", 0)),
        tx_layers=int(config["tx_layers"]),
        tx_heads=int(config["tx_heads"]),
        dropout=float(config["dropout"]),
        eps=float(config.get("eps", 1e-4)),
        predict_u=True,
        predict_log_diag=(config.get("target_mode") == "log_diag"),
        predict_qr_log_diag=(config.get("target_mode") == "qr_log_diag"),
        arch=config.get("arch", "transformer"),
        seq_pool=config.get("seq_pool", "mean"),
        patch_len=int(config.get("patch_len", 12)),
        patch_stride=int(config.get("patch_stride", 6)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = MPCDataset(data_root, drop_duplicate_seq_feature=True,
                    target_scale=config.get("target_scale", "none"))
    ds.set_normalization_stats(*_load_normalization(run_dir))
    _, val_idx = split_indices(len(ds), float(config.get("val_frac", 0.1)),
                               int(config.get("seed", 42)))
    val_idx = val_idx.tolist()
    print(f"[data] {len(val_idx)} validation samples | model: {checkpoint.parent.name}")
    print(f"[out]  {out_dir}")

    # 1a. Target eigenvalues — 4 representative scenarios (2x2 grid)
    print(f"\n[eig] Plotting target eigenvalues for 4 representative scenarios...")
    plot_target_eigenvalues_4scenarios(ds, val_idx, out_dir)

    # 1b. Transformer eigenvalues — single plot, all val samples
    print(f"[eig] Plotting Transformer eigenvalues for up to {args.max_eig} val samples...")
    plot_transformer_eigenvalues(model, ds, val_idx, device, n, m, out_dir, max_eig=args.max_eig)

    # 2. Closed-loop simulation (requires cvxpy)
    if HAS_CVXPY:
        if args.val_indices is not None:
            sim_idx = [val_idx[vi] for vi in args.val_indices]
            print(f"\n[sim] Running closed-loop MPC for {len(sim_idx)} specified samples...")
        else:
            sim_idx = None
            print(f"\n[sim] Running closed-loop MPC simulation ({args.max_sim} samples)...")
        run_sim_and_plot(model, ds, sim_idx if sim_idx else val_idx, device, n, m, out_dir,
                         max_sim=len(sim_idx) if sim_idx else args.max_sim, seed=args.seed)
    else:
        print("\n[sim] Skipped — install cvxpy to enable closed-loop simulation")

    print(f"\n[done] All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
