from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from create_mpc_dataset import (  # noqa: E402
    MPCConfig,
    ParametricMPCController,
    ScenarioConfig,
    compute_metrics,
    generate_disturbance,
    generate_reference,
    lhs_candidates,
    simulate_closed_loop,
    theta_to_qr,
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _scenario_from_meta(meta: dict[str, Any]) -> ScenarioConfig:
    s = meta["scenario"]
    return ScenarioConfig(
        reference_type=s["reference_type"],
        ref_amplitude=float(s["ref_amplitude"]),
        ref_secondary_amplitude=float(s["ref_secondary_amplitude"]),
        ref_switch_step=int(s["ref_switch_step"]),
        x0=[float(v) for v in s["x0"]],
        disturbance_step=int(s["disturbance_step"]),
        disturbance_amplitude=float(s["disturbance_amplitude"]),
        measurement_noise_std=float(s["measurement_noise_std"]),
    )


def band_penalty(value: float, low: float, high: float, scale: float) -> float:
    if value < low:
        return ((low - value) / max(scale, 1e-9)) ** 2
    if value > high:
        return ((value - high) / max(scale, 1e-9)) ** 2
    return 0.0


def spec_cost(
    metrics: dict[str, float],
    sim: dict[str, Any],
    reference: np.ndarray,
    theta: np.ndarray,
    mpc_cfg: MPCConfig,
    args: argparse.Namespace,
) -> tuple[float, dict[str, float]]:
    t_total = max(len(reference) * mpc_cfg.Ts, 1e-9)
    ref_scale = max(float(np.max(np.abs(reference))), 1.0)
    u = np.asarray(sim["u"], dtype=float)

    iae_norm = metrics["iae"] / (t_total * ref_scale)
    ise_norm = metrics["ise"] / (t_total * ref_scale * ref_scale)
    energy_norm = metrics["control_energy"] / (t_total * max(mpc_cfg.u_max ** 2, 1e-9))
    variation_norm = metrics["control_variation"] / (t_total * max(mpc_cfg.du_max ** 2, 1e-9))
    settling_norm = metrics["settling_time"] / t_total
    saturation_fraction = float(np.mean(np.abs(u) >= args.saturation_ratio * mpc_cfg.u_max)) if u.size else 1.0

    p_tracking = args.w_tracking * (iae_norm + ise_norm)
    p_overshoot = args.w_overshoot * max(0.0, metrics["overshoot"] - args.overshoot_max) ** 2
    p_settling = args.w_settling * band_penalty(settling_norm, args.settling_min_norm, args.settling_max_norm, args.settling_scale)
    p_energy = args.w_energy * band_penalty(energy_norm, args.energy_min_norm, args.energy_max_norm, args.energy_scale)
    p_variation = args.w_variation * band_penalty(variation_norm, args.variation_min_norm, args.variation_max_norm, args.variation_scale)
    p_saturation = args.w_saturation * saturation_fraction

    # Avoid both scale collapse and scale explosion, but softly.
    p_theta_band = args.w_theta_band * (
        band_penalty(float(theta[0]), args.q_pref_min_exp, args.q_pref_max_exp, args.theta_scale)
        + band_penalty(float(theta[1]), args.q_pref_min_exp, args.q_pref_max_exp, args.theta_scale)
        + band_penalty(float(theta[2]), args.r_pref_min_exp, args.r_pref_max_exp, args.theta_scale)
    )

    parts = {
        "iae_norm": iae_norm,
        "ise_norm": ise_norm,
        "energy_norm": energy_norm,
        "variation_norm": variation_norm,
        "settling_norm": settling_norm,
        "saturation_fraction": saturation_fraction,
        "p_tracking": p_tracking,
        "p_overshoot": p_overshoot,
        "p_settling": p_settling,
        "p_energy": p_energy,
        "p_variation": p_variation,
        "p_saturation": p_saturation,
        "p_theta_band": p_theta_band,
    }
    return float(sum(parts[k] for k in parts if k.startswith("p_"))), parts


def evaluate(
    theta: np.ndarray,
    controller: ParametricMPCController,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    E: np.ndarray,
    reference: np.ndarray,
    disturbance: np.ndarray,
    x0: np.ndarray,
    mpc_cfg: MPCConfig,
    noise_std: float,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    q_diag, r_diag = theta_to_qr(theta)
    sim = simulate_closed_loop(
        controller=controller,
        A=A,
        B=B,
        C=C,
        E=E,
        q_diag=q_diag,
        r_diag=r_diag,
        reference=reference,
        disturbance=disturbance,
        x0=x0,
        mpc_cfg=mpc_cfg,
        noise_std=noise_std,
        rng=np.random.default_rng(seed),
    )
    if not sim["feasible"]:
        return {"feasible": False, "cost": 1e6}
    metrics = compute_metrics(sim["y"], sim["u"], reference, mpc_cfg.Ts)
    cost, parts = spec_cost(metrics, sim, reference, theta, mpc_cfg, args)
    return {"feasible": True, "cost": cost, "metrics": metrics, "parts": parts}


def make_candidates(args: argparse.Namespace, seed: int, old_theta: np.ndarray) -> np.ndarray:
    lower = np.array([args.q_min_exp, args.q_min_exp, args.r_min_exp], dtype=float)
    upper = np.array([args.q_max_exp, args.q_max_exp, args.r_max_exp], dtype=float)
    lhs = lhs_candidates(args.n_candidates, lower, upper, seed)
    anchors = np.array(
        [
            old_theta,
            [args.q_pref_min_exp, args.q_pref_min_exp, args.r_pref_min_exp],
            [(args.q_pref_min_exp + args.q_pref_max_exp) / 2, (args.q_pref_min_exp + args.q_pref_max_exp) / 2, (args.r_pref_min_exp + args.r_pref_max_exp) / 2],
            [args.q_pref_max_exp, args.q_pref_max_exp, args.r_pref_max_exp],
            [args.q_min_exp, args.q_min_exp, args.r_min_exp],
        ],
        dtype=float,
    )
    return np.vstack([anchors, lhs])


def _plot(theta_old: np.ndarray, theta_new: np.ndarray, out_dir: Path) -> None:
    names = ["log10(Q11)", "log10(Q22)", "log10(R11)"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, ax in enumerate(axes):
        ax.hist(theta_old[:, i], bins=18, alpha=0.55, label="old", color="#94a3b8")
        ax.hist(theta_new[:, i], bins=18, alpha=0.65, label="spec-based", color="#2563eb")
        ax.set_title(names[i])
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "old_vs_spec_theta_histograms.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dirs = sorted((input_root / "samples").glob("sample_*"))[: args.max_samples]
    rng = np.random.default_rng(args.random_seed)

    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for sample_dir in sample_dirs:
        meta = _load_json(sample_dir / "meta.json")
        model = meta["model"]
        constraints = meta["constraints"]
        mpc_cfg = MPCConfig(
            Ts=float(model.get("Ts", 0.02)),
            horizon=int(args.horizon or constraints.get("horizon", 15)),
            sim_steps=int(args.sim_steps or constraints.get("sim_steps", 120)),
            u_min=float(constraints.get("u_min", -24.0)),
            u_max=float(constraints.get("u_max", 24.0)),
            du_min=float(constraints.get("du_min", -4.0)),
            du_max=float(constraints.get("du_max", 4.0)),
            solver="OSQP",
            solver_max_iter=args.solver_max_iter,
        )
        A = np.asarray(model["A_d"], dtype=float)
        B = np.asarray(model["B_d"], dtype=float)
        C = np.asarray(model["C_d"], dtype=float)
        E = np.asarray(model["E_d"], dtype=float)
        scfg = _scenario_from_meta(meta)
        reference = generate_reference(scfg, mpc_cfg.sim_steps, mpc_cfg.Ts)
        disturbance = generate_disturbance(scfg, mpc_cfg.sim_steps)
        x0 = np.asarray(scfg.x0, dtype=float)
        controller = ParametricMPCController(A=A, B=B, C=C, E=E, mpc_cfg=mpc_cfg)

        old_theta = np.asarray(meta["optimization"]["best_theta"], dtype=float)
        seed = int(rng.integers(0, 1_000_000))
        candidates = make_candidates(args, seed, old_theta)

        old_cost = None
        best = None
        for cand_idx, theta in enumerate(candidates):
            res = evaluate(theta, controller, A, B, C, E, reference, disturbance, x0, mpc_cfg, scfg.measurement_noise_std, seed, args)
            if cand_idx == 0:
                old_cost = float(res["cost"])
            if best is None or res["cost"] < best["cost"]:
                best = {"theta": np.asarray(theta, dtype=float), **res}
            parts = res.get("parts", {})
            candidate_rows.append({
                "sample": sample_dir.name,
                "candidate_idx": cand_idx,
                "theta_q1": float(theta[0]),
                "theta_q2": float(theta[1]),
                "theta_r": float(theta[2]),
                "cost": float(res["cost"]),
                "feasible": bool(res["feasible"]),
                **{k: float(v) for k, v in parts.items()},
            })

        assert best is not None and old_cost is not None
        metrics = best.get("metrics", {})
        parts = best.get("parts", {})
        row = {
            "sample": sample_dir.name,
            "old_theta_q1": float(old_theta[0]),
            "old_theta_q2": float(old_theta[1]),
            "old_theta_r": float(old_theta[2]),
            "new_theta_q1": float(best["theta"][0]),
            "new_theta_q2": float(best["theta"][1]),
            "new_theta_r": float(best["theta"][2]),
            "old_spec_cost": old_cost,
            "new_spec_cost": float(best["cost"]),
            "improvement_ratio": float(old_cost / max(float(best["cost"]), 1e-12)),
            "iae": float(metrics.get("iae", np.nan)),
            "ise": float(metrics.get("ise", np.nan)),
            "control_energy": float(metrics.get("control_energy", np.nan)),
            "control_variation": float(metrics.get("control_variation", np.nan)),
            "overshoot": float(metrics.get("overshoot", np.nan)),
            "settling_time": float(metrics.get("settling_time", np.nan)),
            **{k: float(v) for k, v in parts.items()},
        }
        rows.append(row)
        print(
            f"[SPEC] {sample_dir.name} old=({old_theta[0]:.2f},{old_theta[1]:.2f},{old_theta[2]:.2f}) "
            f"new=({best['theta'][0]:.2f},{best['theta'][1]:.2f},{best['theta'][2]:.2f}) "
            f"cost={old_cost:.3f}->{best['cost']:.3f}",
            flush=True,
        )

    _write_csv(out_dir / "spec_best_per_sample.csv", rows)
    _write_csv(out_dir / "spec_all_candidates.csv", candidate_rows)
    old_theta = np.array([[r["old_theta_q1"], r["old_theta_q2"], r["old_theta_r"]] for r in rows], dtype=float)
    new_theta = np.array([[r["new_theta_q1"], r["new_theta_q2"], r["new_theta_r"]] for r in rows], dtype=float)
    _plot(old_theta, new_theta, out_dir)

    summary = {
        "num_samples": len(rows),
        "new_theta_median": {
            "q1": float(np.median(new_theta[:, 0])),
            "q2": float(np.median(new_theta[:, 1])),
            "r": float(np.median(new_theta[:, 2])),
        },
        "new_theta_std": {
            "q1": float(np.std(new_theta[:, 0])),
            "q2": float(np.std(new_theta[:, 1])),
            "r": float(np.std(new_theta[:, 2])),
        },
        "new_at_same_anchor_pct": float(np.mean(np.all(np.isclose(new_theta, new_theta[0], atol=1e-6), axis=1)) * 100.0),
        "median_improvement_ratio": float(np.median([r["improvement_ratio"] for r in rows])),
    }
    (out_dir / "spec_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "README_spec_recalibration.md").write_text(
        f"""# Spec-Based Recalibration

Samples: **{len(rows)}**

Costul penalizeaza abaterea de la benzi dorite pentru settling, energie control si variatie, plus tracking si overshoot.

Median theta nou:
- q1: **{summary['new_theta_median']['q1']:.2f}**
- q2: **{summary['new_theta_median']['q2']:.2f}**
- r: **{summary['new_theta_median']['r']:.2f}**

Std theta nou:
- q1: **{summary['new_theta_std']['q1']:.2f}**
- q2: **{summary['new_theta_std']['q2']:.2f}**
- r: **{summary['new_theta_std']['r']:.2f}**

Imbunatatire mediana cost: **{summary['median_improvement_ratio']:.2f}x**

Fisiere:
- `spec_best_per_sample.csv`
- `spec_all_candidates.csv`
- `old_vs_spec_theta_histograms.png`
""",
        encoding="utf-8",
    )
    print(f"[done] spec recalibration saved to {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spec-based recalibration for existing MPC scenarios.")
    p.add_argument("--input_root", default="MPC_dataset/mpc_qr_dataset/mpc_pqr_dataset_realfit_3000_merged")
    p.add_argument("--out_dir", default="reports/recalibrate_existing_specs")
    p.add_argument("--max_samples", type=int, default=30)
    p.add_argument("--n_candidates", type=int, default=48)
    p.add_argument("--random_seed", type=int, default=7070)
    p.add_argument("--q_min_exp", type=float, default=-8.0)
    p.add_argument("--q_max_exp", type=float, default=2.0)
    p.add_argument("--r_min_exp", type=float, default=-9.0)
    p.add_argument("--r_max_exp", type=float, default=1.0)
    p.add_argument("--q_pref_min_exp", type=float, default=-5.5)
    p.add_argument("--q_pref_max_exp", type=float, default=-2.0)
    p.add_argument("--r_pref_min_exp", type=float, default=-6.5)
    p.add_argument("--r_pref_max_exp", type=float, default=-3.0)
    p.add_argument("--theta_scale", type=float, default=1.0)
    p.add_argument("--settling_min_norm", type=float, default=0.15)
    p.add_argument("--settling_max_norm", type=float, default=0.85)
    p.add_argument("--settling_scale", type=float, default=0.25)
    p.add_argument("--energy_min_norm", type=float, default=1e-4)
    p.add_argument("--energy_max_norm", type=float, default=2e-3)
    p.add_argument("--energy_scale", type=float, default=8e-4)
    p.add_argument("--variation_min_norm", type=float, default=1e-5)
    p.add_argument("--variation_max_norm", type=float, default=2e-3)
    p.add_argument("--variation_scale", type=float, default=8e-4)
    p.add_argument("--overshoot_max", type=float, default=0.08)
    p.add_argument("--w_tracking", type=float, default=1.0)
    p.add_argument("--w_overshoot", type=float, default=2.0)
    p.add_argument("--w_settling", type=float, default=1.0)
    p.add_argument("--w_energy", type=float, default=1.0)
    p.add_argument("--w_variation", type=float, default=0.5)
    p.add_argument("--w_saturation", type=float, default=2.0)
    p.add_argument("--w_theta_band", type=float, default=0.15)
    p.add_argument("--saturation_ratio", type=float, default=0.98)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--sim_steps", type=int, default=60)
    p.add_argument("--solver_max_iter", type=int, default=2000)
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
