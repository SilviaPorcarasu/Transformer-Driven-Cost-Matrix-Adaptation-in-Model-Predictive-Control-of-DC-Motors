"""
Closed-loop MPC evaluation comparing three variants of (P, Q, R):
  1. Baseline  — ground-truth PQR from optimization (oracle)
  2. Transformer — model predictions
  3. Transformer + MinT — reconciled via Lagrange projection onto DARE manifold

Produces:
  - Table 3 (aggregate control metrics)
  - Table 4 (breakdown by difficulty)
  - Grafic 4 (trajectory overlays for representative samples)
  - Grafic 5 (boxplots IAE_norm / final_error_norm)
  - Grafic 6 (scatter Transformer vs MinT)
  - JSON + CSV dumps
"""
from __future__ import annotations

import os
import sys
import json
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_eval import (
    HybridControllerModel, MPCDataset,
    log_diag_to_matrix, denormalize_log,
    mint_projection, riccati_residual,
)


# ============================================================
# MPC controller (diagonal P, Q, R) — mirrors create_mpc_dataset.py
# ============================================================

class ParametricMPC:
    def __init__(self, A, B, C, E, u_min, u_max, du_min, du_max, horizon):
        self.A, self.B, self.C, self.E = A, B, C, E
        n = A.shape[0]; m = B.shape[1]; N = horizon
        self.n, self.m, self.N = n, m, N
        self.u_min, self.u_max, self.du_min, self.du_max = u_min, u_max, du_min, du_max

        self.x = cp.Variable((n, N + 1))
        self.u = cp.Variable((m, N))
        self.x0_p = cp.Parameter(n)
        self.u_prev_p = cp.Parameter(m)
        self.d_p = cp.Parameter(N)
        self.x_ref_p = cp.Parameter((n, N + 1))
        self.sqrt_p_p = cp.Parameter(n, nonneg=True)
        self.sqrt_q_p = cp.Parameter(n, nonneg=True)
        self.sqrt_r_p = cp.Parameter(m, nonneg=True)

        cost = 0.0
        cons = [self.x[:, 0] == self.x0_p]
        for k in range(N):
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
            cost += cp.sum_squares(cp.multiply(self.sqrt_q_p, x_err))
            cost += cp.sum_squares(cp.multiply(self.sqrt_r_p, self.u[:, k]))
        x_err_N = self.x[:, N] - self.x_ref_p[:, N]
        cost += cp.sum_squares(cp.multiply(self.sqrt_p_p, x_err_N))

        self.problem = cp.Problem(cp.Minimize(cost), cons)

    def solve(self, x0, u_prev, x_ref_hor, d_hor, p_diag, q_diag, r_diag):
        self.x0_p.value = np.asarray(x0, dtype=float)
        self.u_prev_p.value = np.array([u_prev], dtype=float)
        self.x_ref_p.value = np.asarray(x_ref_hor, dtype=float)
        self.d_p.value = np.asarray(d_hor, dtype=float)
        self.sqrt_p_p.value = np.sqrt(np.maximum(p_diag, 1e-12))
        self.sqrt_q_p.value = np.sqrt(np.maximum(q_diag, 1e-12))
        self.sqrt_r_p.value = np.sqrt(np.maximum(r_diag, 1e-12))
        try:
            self.problem.solve(solver=cp.OSQP, warm_start=True, verbose=False, max_iter=4000)
        except Exception:
            return None, "solver_exception"
        if self.problem.status not in ("optimal", "optimal_inaccurate"):
            return None, self.problem.status
        return float(self.u[:, 0].value.item()), self.problem.status


def simulate_closed_loop(mpc, A, B, C, E, p_diag, q_diag, r_diag,
                         x_ref_full, reference, disturbance, x0, sim_steps,
                         horizon, noise_std, rng):
    n = A.shape[0]
    x = np.zeros((sim_steps + 1, n))
    y = np.zeros(sim_steps)
    u = np.zeros(sim_steps)
    e = np.zeros(sim_steps)
    x[0] = x0
    u_prev = 0.0

    for k in range(sim_steps):
        y[k] = float((C @ x[k]).item())
        y_meas = y[k] + rng.normal(0.0, noise_std)

        d_hor = np.array([disturbance[min(k + j, sim_steps - 1)] for j in range(horizon)])
        x_ref_hor = np.column_stack([x_ref_full[:, min(k + j, sim_steps - 1)]
                                     for j in range(horizon + 1)])
        x_meas = x[k].copy()
        x_meas[1] = y_meas

        uk, status = mpc.solve(x_meas, u_prev, x_ref_hor, d_hor, p_diag, q_diag, r_diag)
        if uk is None:
            return {"feasible": False, "y": y[:k], "u": u[:k], "e": e[:k], "status": status}

        u[k] = uk
        e[k] = reference[k] - y[k]
        x[k + 1] = A @ x[k] + B.flatten() * u[k] + E.flatten() * disturbance[k]
        u_prev = u[k]
        if np.any(np.abs(x[k + 1]) > 1e4) or abs(y[k]) > 1e4:
            return {"feasible": False, "y": y[:k + 1], "u": u[:k + 1], "e": e[:k + 1],
                    "status": "diverged"}

    return {"feasible": True, "y": y, "u": u, "e": e, "status": "optimal"}


def compute_control_metrics(y, u, reference, Ts, u_min, u_max):
    if len(y) == 0:
        return None
    e = reference[:len(y)] - y
    ref_scale = max(1.0, float(np.max(np.abs(reference[:len(y)]))))

    iae = float(np.sum(np.abs(e)) * Ts)
    ise = float(np.sum(e ** 2) * Ts)
    iae_norm = iae / ref_scale
    ise_norm = ise / (ref_scale ** 2)
    rmse_norm = float(np.sqrt(np.mean(e ** 2)) / ref_scale)

    ce = float(np.sum(u ** 2) * Ts)
    cv = float(np.sum(np.diff(u) ** 2) * Ts) if len(u) > 1 else 0.0

    eps_sat = 1e-3
    sat_mask = (np.abs(u - u_max) < eps_sat) | (np.abs(u - u_min) < eps_sat)
    sat_ratio = float(np.mean(sat_mask)) if len(u) else 0.0

    final_ref = float(reference[len(y) - 1])
    final_err = float(abs(e[-1]))
    final_err_norm = final_err / max(1.0, abs(final_ref))

    overshoot = float(max(0.0, (np.max(y) - final_ref) / abs(final_ref))) if abs(final_ref) > 1e-8 else 0.0

    settling = float(len(y) * Ts)
    band = 0.02 * max(1.0, abs(final_ref))
    for k in range(len(y)):
        if np.all(np.abs(y[k:] - final_ref) <= band):
            settling = float(k * Ts)
            break

    meta_cost = (3.0 * iae_norm + 3.0 * ise_norm + 2.5 * final_err_norm
                 + 1.0 * sat_ratio + 0.01 * ce + 0.02 * cv
                 + 2.0 * overshoot + 0.1 * settling)
    quality = max(0.0, 1.0 - 0.50 * min(final_err_norm, 1.0)
                       - 0.30 * min(sat_ratio, 1.0)
                       - 0.20 * min(overshoot, 1.0))

    return {
        "iae_norm": iae_norm, "ise_norm": ise_norm, "rmse_norm": rmse_norm,
        "final_error_norm": final_err_norm,
        "sat_ratio": sat_ratio, "overshoot": overshoot, "settling_time": settling,
        "control_energy": ce, "control_variation": cv,
        "meta_cost": float(meta_cost), "quality_score": float(quality),
    }


# ============================================================
# Main
# ============================================================

def agg_stats(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)])
    if len(v) == 0:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan"), "n": 0}
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "n": int(len(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--max_samples", type=int, default=120,
                    help="cap closed-loop sims for speed (default 120)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device(args.device)
    if args.out_dir is None:
        args.out_dir = os.path.join(os.path.dirname(args.checkpoint), "eval_control")
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load checkpoint ──
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["args"]
    stats_in = ckpt.get("input_stats", {})
    stats_log = ckpt.get("stats", {})

    d_seq = int(cfg["d_seq"])
    d_static = int(cfg["d_static"])
    n = int(cfg["n"]); m = int(cfg["m"])

    model = HybridControllerModel(
        d_seq=d_seq, d_static=d_static,
        d_model=int(cfg["d_model"]), n=n, m=m,
        tx_layers=int(cfg["tx_layers"]), tx_heads=int(cfg["tx_heads"]),
        dropout=float(cfg["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mean_P = torch.tensor(stats_log["mean_P"], dtype=torch.float32, device=device)
    std_P = torch.tensor(stats_log["std_P"], dtype=torch.float32, device=device)
    mean_Q = torch.tensor(stats_log["mean_Q"], dtype=torch.float32, device=device)
    std_Q = torch.tensor(stats_log["std_Q"], dtype=torch.float32, device=device)
    mean_R = torch.tensor(stats_log["mean_R"], dtype=torch.float32, device=device)
    std_R = torch.tensor(stats_log["std_R"], dtype=torch.float32, device=device)

    # ── Dataset (val split) ──
    ds = MPCDataset(root_dir=args.data_root,
                    drop_duplicate_seq_feature=(d_seq == 4),
                    duplicate_seq_feature_idx=3)
    ds.set_normalization_stats(stats_in.get("seq_mean"), stats_in.get("seq_std"),
                                stats_in.get("static_mean"), stats_in.get("static_std"))

    rng = np.random.RandomState(args.seed)
    n_total = len(ds)
    n_val = int(float(cfg["val_frac"]) * n_total)
    idxs = list(range(n_total))
    rng.shuffle(idxs)
    val_idx = idxs[:n_val][:args.max_samples]
    print(f"[data] evaluating {len(val_idx)} / {n_val} val samples (capped for speed)")

    # ── Run closed-loop for each sample ──
    results_rows = []
    traj_save = []  # for grafic 4
    mpc_cache = {}  # cache controller per (A,B,C,E,horizon)

    for i, si in enumerate(val_idx):
        item = ds[si]
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
        noise_std = float(meta["scenario"]["measurement_noise_std"])
        difficulty = meta["scenario"].get("difficulty", "unknown")
        ref_type = meta["scenario"]["reference_type"]

        # Ground truth PQR
        p_gt = np.array(meta["target"]["P_diag"])
        q_gt = np.array(meta["target"]["Q_diag"])
        r_gt = np.array(meta["target"]["R_diag"])

        # Transformer prediction
        with torch.no_grad():
            ts = item["tokens_seq"].unsqueeze(0).to(device)
            tst = item["tokens_static"].unsqueeze(0).to(device)
            p_log_n, q_log_n, r_log_n, _ = model(ts, tst)
            p_log = denormalize_log(p_log_n, mean_P, std_P)
            q_log = denormalize_log(q_log_n, mean_Q, std_Q)
            r_log = denormalize_log(r_log_n, mean_R, std_R)
            P_pred = log_diag_to_matrix(p_log)
            Q_pred = log_diag_to_matrix(q_log)
            R_pred = log_diag_to_matrix(r_log)

        p_pred = torch.pow(10.0, p_log).squeeze(0).cpu().numpy()
        q_pred = torch.pow(10.0, q_log).squeeze(0).cpu().numpy()
        r_pred = torch.pow(10.0, r_log).squeeze(0).cpu().numpy()

        # MinT projection
        A_t = torch.tensor(A_d[None]).float()
        B_t = torch.tensor(B_d[None]).float()
        P_proj, Q_proj, R_proj = mint_projection(P_pred.cpu(), Q_pred.cpu(), R_pred.cpu(),
                                                  A_t, B_t, n_iter=3)
        p_mint = np.maximum(np.diag(P_proj.squeeze(0).numpy()), 1e-8)
        q_mint = np.maximum(np.diag(Q_proj.squeeze(0).numpy()), 1e-8)
        r_mint = np.maximum(np.diag(R_proj.squeeze(0).numpy()), 1e-8)

        # Controller cache
        key = (tuple(A_d.flatten().round(6)), horizon)
        if key not in mpc_cache:
            mpc_cache[key] = ParametricMPC(A_d, B_d, C_d, E_d,
                                           u_min, u_max, du_min, du_max, horizon)
        mpc = mpc_cache[key]

        # Three simulations
        sim_seed = 100000 + si
        row = {"sample_idx": int(si), "difficulty": difficulty, "ref_type": ref_type}
        sims_for_plot = {}

        for name, (pd_, qd_, rd_) in [
            ("baseline", (p_gt, q_gt, r_gt)),
            ("tfmr", (p_pred, q_pred, r_pred)),
            ("mint", (p_mint, q_mint, r_mint)),
        ]:
            rng_sim = np.random.default_rng(sim_seed)
            sim = simulate_closed_loop(mpc, A_d, B_d, C_d, E_d, pd_, qd_, rd_,
                                       x_ref_full, reference, disturbance, x0,
                                       sim_steps, horizon, noise_std, rng_sim)
            if sim["feasible"]:
                metrics = compute_control_metrics(sim["y"], sim["u"], reference, Ts, u_min, u_max)
                row[f"{name}_feasible"] = True
                for k, v in metrics.items():
                    row[f"{name}_{k}"] = v
                sims_for_plot[name] = sim
            else:
                row[f"{name}_feasible"] = False
                sims_for_plot[name] = sim

        # Riccati residuals
        row["ric_before_mint"] = float(riccati_residual(
            P_pred.cpu(), Q_pred.cpu(), R_pred.cpu(), A_t, B_t).item())
        row["ric_after_mint"] = float(riccati_residual(
            P_proj, Q_proj, R_proj, A_t, B_t).item())

        results_rows.append(row)
        # Save ~6 representative trajectories
        if len(traj_save) < 6 and all(row.get(f"{n}_feasible") for n in ["baseline", "tfmr", "mint"]):
            traj_save.append({
                "sample_idx": int(si), "difficulty": difficulty, "ref_type": ref_type,
                "reference": reference, "Ts": Ts, "sims": sims_for_plot,
            })

        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1}/{len(val_idx)}")

    # ── Aggregate Tabel 3 ──
    metrics_keys = ["iae_norm", "ise_norm", "rmse_norm", "final_error_norm",
                    "sat_ratio", "overshoot", "settling_time",
                    "control_energy", "control_variation",
                    "meta_cost", "quality_score"]
    variants = ["baseline", "tfmr", "mint"]

    table3 = {}
    for mk in metrics_keys:
        table3[mk] = {v: agg_stats([r.get(f"{v}_{mk}") for r in results_rows
                                    if r.get(f"{v}_feasible", False)]) for v in variants}

    # Feasibility rates
    feas_rate = {v: float(np.mean([r.get(f"{v}_feasible", False) for r in results_rows]))
                 for v in variants}

    # ── Tabel 4: breakdown by difficulty ──
    diff_buckets = sorted(set(r["difficulty"] for r in results_rows))
    table4 = {}
    for diff in diff_buckets:
        subset = [r for r in results_rows if r["difficulty"] == diff]
        table4[diff] = {
            "n_samples": len(subset),
            "ric_before_med": agg_stats([r["ric_before_mint"] for r in subset])["median"],
            "ric_after_med": agg_stats([r["ric_after_mint"] for r in subset])["median"],
        }
        for v in variants:
            for mk in ["iae_norm", "final_error_norm", "quality_score"]:
                table4[diff][f"{v}_{mk}_median"] = agg_stats(
                    [r.get(f"{v}_{mk}") for r in subset if r.get(f"{v}_feasible", False)]
                )["median"]
        samples_help = sum(1 for r in subset if r.get("tfmr_feasible") and r.get("mint_feasible")
                           and r.get("mint_iae_norm", np.inf) < r.get("tfmr_iae_norm", np.inf))
        total_both = sum(1 for r in subset if r.get("tfmr_feasible") and r.get("mint_feasible"))
        table4[diff]["pct_mint_helps"] = (100.0 * samples_help / total_both) if total_both else 0.0

    # ── Print ──
    print("\n" + "=" * 90)
    print("TABEL 3 — Control performance (closed-loop, {} samples)".format(len(results_rows)))
    print("=" * 90)
    header = f"{'Metric':22s} | {'Baseline (oracle)':>22s} | {'Transformer':>22s} | {'Transformer+MinT':>22s}"
    print(header)
    print("-" * len(header))
    for mk in metrics_keys:
        bs = table3[mk]["baseline"]; ts = table3[mk]["tfmr"]; ms = table3[mk]["mint"]
        print(f"{mk:22s} | med={bs['median']:8.4f} mean={bs['mean']:8.4f} | "
              f"med={ts['median']:8.4f} mean={ts['mean']:8.4f} | "
              f"med={ms['median']:8.4f} mean={ms['mean']:8.4f}")
    print("-" * len(header))
    print(f"{'feasibility rate':22s} | {feas_rate['baseline']*100:>20.1f}% | "
          f"{feas_rate['tfmr']*100:>20.1f}% | {feas_rate['mint']*100:>20.1f}%")

    print("\n" + "=" * 90)
    print("TABEL 4 — Breakdown by scenario difficulty")
    print("=" * 90)
    for diff, row in table4.items():
        print(f"\n  [{diff}]  n={row['n_samples']}")
        print(f"    Riccati residual   median: before={row['ric_before_med']:.2f}  "
              f"after={row['ric_after_med']:.2f}")
        for v in variants:
            print(f"    {v:<12s}  IAE_norm_med={row.get(f'{v}_iae_norm_median', float('nan')):.4f}  "
                  f"final_err_med={row.get(f'{v}_final_error_norm_median', float('nan')):.4f}  "
                  f"quality_med={row.get(f'{v}_quality_score_median', float('nan')):.4f}")
        print(f"    → MinT helps vs Transformer (lower IAE): {row['pct_mint_helps']:.1f}%")

    # ── Save CSV/JSON ──
    with open(os.path.join(args.out_dir, "per_sample_control.csv"), "w", newline="") as f:
        if results_rows:
            all_keys = set()
            for r in results_rows:
                all_keys.update(r.keys())
            w = csv.DictWriter(f, fieldnames=sorted(all_keys))
            w.writeheader()
            w.writerows(results_rows)

    with open(os.path.join(args.out_dir, "control_summary.json"), "w") as fp:
        json.dump({
            "checkpoint": args.checkpoint,
            "epoch": ckpt.get("epoch"),
            "n_samples": len(results_rows),
            "feasibility_rate": feas_rate,
            "table3_aggregate": table3,
            "table4_by_difficulty": table4,
        }, fp, indent=2)

    # ── Grafic 4: trajectory overlays ──
    if traj_save:
        fig, axes = plt.subplots(len(traj_save), 2, figsize=(14, 3 * len(traj_save)))
        if len(traj_save) == 1:
            axes = axes[None, :]
        for i, tr in enumerate(traj_save):
            Ts = tr["Ts"]
            ref = tr["reference"]
            t = np.arange(len(ref)) * Ts

            ax = axes[i, 0]
            ax.plot(t, ref, "k-", label="reference", linewidth=2, alpha=0.7)
            for name, color, label in [
                ("baseline", "tab:green", "Baseline (oracle)"),
                ("tfmr", "tab:blue", "Transformer"),
                ("mint", "tab:red", "Transformer+MinT"),
            ]:
                s = tr["sims"][name]
                if s["feasible"]:
                    ax.plot(np.arange(len(s["y"])) * Ts, s["y"], color=color,
                            label=label, linewidth=1.5, alpha=0.85)
            ax.set_title(f"Sample {tr['sample_idx']}  |  {tr['difficulty']}  |  {tr['ref_type']}")
            ax.set_xlabel("Time [s]"); ax.set_ylabel("Output y")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

            ax = axes[i, 1]
            for name, color, label in [
                ("baseline", "tab:green", "Baseline"),
                ("tfmr", "tab:blue", "Transformer"),
                ("mint", "tab:red", "Transformer+MinT"),
            ]:
                s = tr["sims"][name]
                if s["feasible"]:
                    ax.plot(np.arange(len(s["u"])) * Ts, s["u"], color=color,
                            label=label, linewidth=1.2, alpha=0.85)
            ax.axhline(24, color="gray", linestyle=":", alpha=0.5)
            ax.axhline(-24, color="gray", linestyle=":", alpha=0.5)
            ax.set_title("Control input u(t)")
            ax.set_xlabel("Time [s]"); ax.set_ylabel("u")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "grafic4_trajectories.png"), dpi=120)
        plt.close(fig)

    # ── Grafic 5: boxplots ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, mk, title in zip(
        axes,
        ["iae_norm", "final_error_norm", "quality_score"],
        ["IAE (normalized)", "Final error (normalized)", "Quality score"],
    ):
        data = []
        labels = []
        for v, lab in [("baseline", "Baseline"), ("tfmr", "Transformer"), ("mint", "Transformer+MinT")]:
            vals = [r.get(f"{v}_{mk}") for r in results_rows if r.get(f"{v}_feasible", False)]
            vals = [x for x in vals if x is not None and np.isfinite(x)]
            data.append(vals); labels.append(lab)
        bp = ax.boxplot(data, labels=labels, showfliers=True, patch_artist=True)
        for patch, color in zip(bp["boxes"], ["tab:green", "tab:blue", "tab:red"]):
            patch.set_facecolor(color); patch.set_alpha(0.6)
        ax.set_title(title); ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=15)
        if mk in ("iae_norm", "final_error_norm"):
            ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "grafic5_boxplots.png"), dpi=120)
    plt.close(fig)

    # ── Grafic 6: scatter Transformer vs MinT ──
    tf_vals, mi_vals = [], []
    for r in results_rows:
        if r.get("tfmr_feasible") and r.get("mint_feasible"):
            tf_vals.append(r["tfmr_iae_norm"]); mi_vals.append(r["mint_iae_norm"])
    if tf_vals:
        fig, ax = plt.subplots(figsize=(7, 7))
        tf_vals = np.array(tf_vals); mi_vals = np.array(mi_vals)
        colors = ["tab:red" if m > t else "tab:green" for t, m in zip(tf_vals, mi_vals)]
        ax.scatter(tf_vals, mi_vals, c=colors, alpha=0.6, s=30)
        lim = [min(tf_vals.min(), mi_vals.min()) * 0.9, max(tf_vals.max(), mi_vals.max()) * 1.1]
        ax.plot(lim, lim, "k--", alpha=0.6, label="y = x")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("IAE_norm — Transformer")
        ax.set_ylabel("IAE_norm — Transformer + MinT")
        ax.set_title("MinT impact per sample (green = MinT helps, red = MinT hurts)")
        help_pct = float(np.mean(mi_vals < tf_vals) * 100)
        ax.text(0.05, 0.95, f"MinT helps: {help_pct:.1f}%  ({(mi_vals < tf_vals).sum()}/{len(tf_vals)})",
                transform=ax.transAxes, fontsize=11, va="top",
                bbox=dict(facecolor="white", alpha=0.8))
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "grafic6_mint_scatter.png"), dpi=120)
        plt.close(fig)

    print(f"\n[done] outputs saved to {args.out_dir}/")
    print(f"  - control_summary.json (Tabel 3 + Tabel 4)")
    print(f"  - per_sample_control.csv")
    print(f"  - grafic4_trajectories.png")
    print(f"  - grafic5_boxplots.png")
    print(f"  - grafic6_mint_scatter.png")


if __name__ == "__main__":
    main()
