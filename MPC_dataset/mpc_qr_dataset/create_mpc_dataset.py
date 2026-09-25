from __future__ import annotations

import csv
import json
import time
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import cont2discrete
from scipy.stats import qmc


# ============================================================
# Configuration dataclasses
# ============================================================

@dataclass
class MotorParams:
    Ra: float
    La: float
    J: float
    b: float
    Kt: float
    Ke: float


@dataclass
class MPCConfig:
    Ts: float = 0.02
    horizon: int = 15
    sim_steps: int = 120
    u_min: float = -24.0
    u_max: float = 24.0
    du_min: float = -4.0
    du_max: float = 4.0
    solver: str = "OSQP"
    solver_max_iter: int = 4000
    # early-stop safety thresholds for obviously bad trajectories
    max_abs_state: float = 1e4
    max_abs_output: float = 1e4


@dataclass
class SearchConfig:
    # Global exploration with Latin Hypercube
    n_lhs_candidates: int = 12
    # Local refinement
    use_powell: bool = True
    powell_maxiter: int = 25
    # Search domain in log10 space
    q_min_exp: float = -3.0
    q_max_exp: float = 3.0
    r_min_exp: float = -4.0
    r_max_exp: float = 2.0


@dataclass
class ScenarioConfig:
    reference_type: str
    ref_amplitude: float
    ref_secondary_amplitude: float
    ref_switch_step: int
    x0: List[float]
    disturbance_step: int
    disturbance_amplitude: float
    measurement_noise_std: float


@dataclass
class DatasetConfig:
    n_models: int = 5
    scenarios_per_model: int = 10
    random_seed: int = 42
    output_dir: str = "mpc_qr_dataset_streaming"
    save_closed_loop_trajectories: bool = True
    save_signals_npz: bool = True
    print_every_sample: bool = True
    plot_first_successful_sample: bool = False
    chunk_jsonl_manifest: bool = False  # optional if you still want a manifest
    resume: bool = False


# ============================================================
# Motor model
# ============================================================

def dc_motor_state_space(params: MotorParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    A = np.array([
        [-params.Ra / params.La, -params.Ke / params.La],
        [params.Kt / params.J,   -params.b / params.J],
    ], dtype=float)

    B = np.array([
        [1.0 / params.La],
        [0.0],
    ], dtype=float)

    C = np.array([[0.0, 1.0]], dtype=float)
    D = np.array([[0.0]], dtype=float)

    return A, B, C, D


def discretize_system(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    Ts: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Ad, Bd, Cd, Dd, _ = cont2discrete((A, B, C, D), Ts, method="zoh")
    return Ad, Bd, Cd, Dd


def disturbance_input_matrix(params: MotorParams, Ts: float) -> np.ndarray:
    A, _, C, D = dc_motor_state_space(params)
    E = np.array([[0.0], [-1.0 / params.J]], dtype=float)
    Ad, Ed, _, _, _ = cont2discrete((A, E, C, D), Ts, method="zoh")
    return Ed


# ============================================================
# Scenario generation
# ============================================================

def generate_reference(scfg: ScenarioConfig, sim_steps: int, Ts: float) -> np.ndarray:
    r = np.zeros(sim_steps, dtype=float)

    if scfg.reference_type == "step":
        r[:] = scfg.ref_amplitude

    elif scfg.reference_type == "multi_step":
        r[:scfg.ref_switch_step] = scfg.ref_amplitude
        r[scfg.ref_switch_step:] = scfg.ref_secondary_amplitude

    elif scfg.reference_type == "sine":
        t = np.arange(sim_steps) * Ts
        freq_hz = 0.15
        offset = 0.5 * (scfg.ref_amplitude + scfg.ref_secondary_amplitude)
        amp = 0.5 * abs(scfg.ref_secondary_amplitude - scfg.ref_amplitude)
        r = offset + amp * np.sin(2.0 * np.pi * freq_hz * t)

    else:
        raise ValueError(f"Unknown reference_type: {scfg.reference_type}")

    return r


def generate_disturbance(scfg: ScenarioConfig, sim_steps: int) -> np.ndarray:
    d = np.zeros(sim_steps, dtype=float)
    d[scfg.disturbance_step:] = scfg.disturbance_amplitude
    return d


def sample_motor_params(nominal: MotorParams, rng: np.random.Generator) -> MotorParams:
    def vary(value: float, low: float, high: float) -> float:
        return float(value * rng.uniform(low, high))

    return MotorParams(
        Ra=vary(nominal.Ra, 0.8, 1.2),
        La=vary(nominal.La, 0.8, 1.2),
        J=vary(nominal.J, 0.7, 1.4),
        b=vary(nominal.b, 0.7, 1.3),
        Kt=vary(nominal.Kt, 0.9, 1.1),
        Ke=vary(nominal.Ke, 0.9, 1.1),
    )


def estimate_second_order_tf_like_features(params: MotorParams) -> Dict[str, float]:
    electrical_time_constant = params.La / max(params.Ra, 1e-8)
    mechanical_time_constant = params.J / max(params.b, 1e-8)
    gain_proxy = params.Kt / max(params.Ra * params.b + params.Kt * params.Ke, 1e-8)

    return {
        "gain_proxy": float(gain_proxy),
        "Te": float(electrical_time_constant),
        "Tm": float(mechanical_time_constant),
    }


def compute_y_max_ss(A_d: np.ndarray, B_d: np.ndarray, C_d: np.ndarray, u_max: float) -> float:
    """Max reachable steady-state output: DC_gain * u_max."""
    try:
        dc_gain = C_d @ np.linalg.inv(np.eye(A_d.shape[0]) - A_d) @ B_d
        return float(abs(dc_gain.item()) * u_max)
    except np.linalg.LinAlgError:
        return float(u_max)  # fallback


def sample_scenario(sim_steps: int, rng: np.random.Generator,
                    y_max_ss: float = 5.0) -> ScenarioConfig:
    ref_type = rng.choice(["step", "multi_step", "sine"]).item()

    # Scale references to [10%, 80%] of max reachable steady-state output
    amp1 = float(rng.uniform(0.1 * y_max_ss, 0.8 * y_max_ss))
    amp2 = float(rng.uniform(0.1 * y_max_ss, 0.8 * y_max_ss))
    switch_step = int(rng.integers(sim_steps // 4, 3 * sim_steps // 4))

    # Scale initial conditions proportionally
    x0 = [
        float(rng.uniform(-0.1 * y_max_ss, 0.1 * y_max_ss)),
        float(rng.uniform(-0.5 * y_max_ss, 0.5 * y_max_ss)),
    ]

    disturbance_step = int(rng.integers(sim_steps // 5, 4 * sim_steps // 5))
    disturbance_amplitude = float(rng.uniform(-0.2, 0.2))
    measurement_noise_std = float(rng.uniform(0.0, 0.5))

    return ScenarioConfig(
        reference_type=ref_type,
        ref_amplitude=amp1,
        ref_secondary_amplitude=amp2,
        ref_switch_step=switch_step,
        x0=x0,
        disturbance_step=disturbance_step,
        disturbance_amplitude=disturbance_amplitude,
        measurement_noise_std=measurement_noise_std,
    )


# ============================================================
# MPC reusable controller
# ============================================================

class ParametricMPCController:
    """
    Builds the CVXPY graph once for a fixed (A,B,C,E,horizon,constraints),
    then updates only parameters at solve time.

    Q and R are assumed diagonal, and are injected through sqrt weights:
      x-cost = ||diag(sqrt_q) x||^2
      u-cost = ||diag(sqrt_r) u||^2
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        E: np.ndarray,
        mpc_cfg: MPCConfig,
    ) -> None:
        self.A = A
        self.B = B
        self.C = C
        self.E = E
        self.cfg = mpc_cfg

        n = A.shape[0]
        m = B.shape[1]
        N = mpc_cfg.horizon

        self.n = n
        self.m = m
        self.N = N

        # Variables
        self.x = cp.Variable((n, N + 1))
        self.u = cp.Variable((m, N))

        # Parameters
        self.x0_p = cp.Parameter(n)
        self.u_prev_p = cp.Parameter(m)
        self.ref_p = cp.Parameter(N)
        self.d_p = cp.Parameter(N)
        self.sqrt_q_p = cp.Parameter(n, nonneg=True)
        self.sqrt_r_p = cp.Parameter(m, nonneg=True)

        cost = 0.0
        constraints = [self.x[:, 0] == self.x0_p]

        for k in range(N):
            constraints += [
                self.x[:, k + 1] == self.A @ self.x[:, k] + self.B @ self.u[:, k] + self.E.flatten() * self.d_p[k],
                self.u[:, k] >= mpc_cfg.u_min,
                self.u[:, k] <= mpc_cfg.u_max,
            ]

            if k == 0:
                constraints += [
                    self.u[:, k] - self.u_prev_p >= mpc_cfg.du_min,
                    self.u[:, k] - self.u_prev_p <= mpc_cfg.du_max,
                ]
            else:
                constraints += [
                    self.u[:, k] - self.u[:, k - 1] >= mpc_cfg.du_min,
                    self.u[:, k] - self.u[:, k - 1] <= mpc_cfg.du_max,
                ]

            # State regularization
            cost += cp.sum_squares(cp.multiply(self.sqrt_q_p, self.x[:, k]))

            # Tracking
            yk = self.C @ self.x[:, k]
            cost += 10.0 * cp.sum_squares(yk - cp.reshape(self.ref_p[k], (1,), order="F"))

            # Control regularization
            cost += cp.sum_squares(cp.multiply(self.sqrt_r_p, self.u[:, k]))

        cost += cp.sum_squares(cp.multiply(self.sqrt_q_p, self.x[:, N]))

        self.problem = cp.Problem(cp.Minimize(cost), constraints)

    def solve_step(
        self,
        x0: np.ndarray,
        u_prev: float,
        ref_horizon: np.ndarray,
        d_horizon: np.ndarray,
        q_diag: np.ndarray,
        r_diag: np.ndarray,
    ) -> Tuple[Optional[float], str]:
        self.x0_p.value = np.asarray(x0, dtype=float)
        self.u_prev_p.value = np.array([u_prev], dtype=float)
        self.ref_p.value = np.asarray(ref_horizon, dtype=float)
        self.d_p.value = np.asarray(d_horizon, dtype=float)

        # sqrt weights for weighted squares
        self.sqrt_q_p.value = np.sqrt(np.maximum(q_diag, 1e-12))
        self.sqrt_r_p.value = np.sqrt(np.maximum(r_diag, 1e-12))

        try:
            if self.cfg.solver.upper() == "OSQP":
                self.problem.solve(
                    solver=cp.OSQP,
                    warm_start=True,
                    verbose=False,
                    max_iter=self.cfg.solver_max_iter,
                )
            else:
                self.problem.solve(
                    solver=self.cfg.solver,
                    warm_start=True,
                    verbose=False,
                )
        except Exception as exc:
            return None, f"solver_exception: {exc}"

        if self.problem.status not in ("optimal", "optimal_inaccurate"):
            return None, self.problem.status

        u0 = float(self.u[:, 0].value.item())
        return u0, self.problem.status


# ============================================================
# Simulation and metrics
# ============================================================

def simulate_closed_loop(
    controller: ParametricMPCController,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    E: np.ndarray,
    q_diag: np.ndarray,
    r_diag: np.ndarray,
    reference: np.ndarray,
    disturbance: np.ndarray,
    x0: np.ndarray,
    mpc_cfg: MPCConfig,
    noise_std: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    n = A.shape[0]
    sim_steps = mpc_cfg.sim_steps

    x = np.zeros((sim_steps + 1, n), dtype=float)
    y = np.zeros(sim_steps, dtype=float)
    y_meas = np.zeros(sim_steps, dtype=float)
    u = np.zeros(sim_steps, dtype=float)
    e = np.zeros(sim_steps, dtype=float)

    x[0] = x0
    u_prev = 0.0

    last_status = "unknown"

    for k in range(sim_steps):
        ref_horizon = np.array(
            [reference[min(k + j, sim_steps - 1)] for j in range(mpc_cfg.horizon)],
            dtype=float,
        )
        d_horizon = np.array(
            [disturbance[min(k + j, sim_steps - 1)] for j in range(mpc_cfg.horizon)],
            dtype=float,
        )

        uk, status = controller.solve_step(
            x0=x[k],
            u_prev=u_prev,
            ref_horizon=ref_horizon,
            d_horizon=d_horizon,
            q_diag=q_diag,
            r_diag=r_diag,
        )
        last_status = status

        if uk is None:
            return {
                "feasible": False,
                "solver_status": status,
                "x": x[:k + 1],
                "y": y[:k],
                "y_meas": y_meas[:k],
                "u": u[:k],
                "e": e[:k],
            }

        u[k] = uk
        y[k] = float((C @ x[k]).item())
        y_meas[k] = y[k] + rng.normal(0.0, noise_std)
        e[k] = reference[k] - y[k]

        x[k + 1] = A @ x[k] + B.flatten() * u[k] + E.flatten() * disturbance[k]
        u_prev = u[k]

        # cheap early failure checks to avoid wasting time
        if np.any(np.abs(x[k + 1]) > mpc_cfg.max_abs_state) or abs(y[k]) > mpc_cfg.max_abs_output:
            return {
                "feasible": False,
                "solver_status": "trajectory_diverged",
                "x": x[:k + 2],
                "y": y[:k + 1],
                "y_meas": y_meas[:k + 1],
                "u": u[:k + 1],
                "e": e[:k + 1],
            }

    return {
        "feasible": True,
        "solver_status": last_status,
        "x": x,
        "y": y,
        "y_meas": y_meas,
        "u": u,
        "e": e,
    }


def compute_metrics(y: np.ndarray, u: np.ndarray, reference: np.ndarray, Ts: float) -> Dict[str, float]:
    e = reference[:len(y)] - y

    iae = float(np.sum(np.abs(e)) * Ts)
    ise = float(np.sum(e ** 2) * Ts)
    control_energy = float(np.sum(u ** 2) * Ts)

    if len(u) > 1:
        delta_u = np.diff(u)
        control_variation = float(np.sum(delta_u ** 2) * Ts)
    else:
        control_variation = 0.0

    final_ref = float(reference[-1])
    if abs(final_ref) > 1e-8:
        overshoot = float(max(0.0, (np.max(y) - final_ref) / abs(final_ref)))
    else:
        overshoot = 0.0

    settling_time = float(len(y) * Ts)
    band = 0.02 * max(1.0, abs(final_ref))
    for k in range(len(y)):
        if np.all(np.abs(y[k:] - final_ref) <= band):
            settling_time = float(k * Ts)
            break

    return {
        "iae": iae,
        "ise": ise,
        "control_energy": control_energy,
        "control_variation": control_variation,
        "overshoot": overshoot,
        "settling_time": settling_time,
    }


def meta_cost(metrics: Dict[str, float]) -> float:
    return (
        3.0 * metrics["iae"]
        + 2.0 * metrics["ise"]
        + 0.05 * metrics["control_energy"]
        + 0.05 * metrics["control_variation"]
        + 10.0 * metrics["overshoot"]
        + 0.5 * metrics["settling_time"]
    )


# ============================================================
# Search: fast global + local refinement
# ============================================================

def theta_to_qr(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=float)
    q1 = 10.0 ** theta[0]
    q2 = 10.0 ** theta[1]
    r1 = 10.0 ** theta[2]
    return np.array([q1, q2], dtype=float), np.array([r1], dtype=float)


def qr_objective(
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
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> float:
    theta = np.asarray(theta, dtype=float)

    if np.any(theta < lower_bounds) or np.any(theta > upper_bounds):
        return 1e8

    q_diag, r_diag = theta_to_qr(theta)
    rng = np.random.default_rng(seed)

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
        rng=rng,
    )

    if not sim["feasible"]:
        return 1e6

    metrics = compute_metrics(sim["y"], sim["u"], reference, mpc_cfg.Ts)
    return meta_cost(metrics)


def lhs_candidates(
    n_candidates: int,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    seed: int,
) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=len(lower_bounds), seed=seed)
    unit_samples = sampler.random(n=n_candidates)
    return qmc.scale(unit_samples, lower_bounds, upper_bounds)


def find_optimal_qr_fast(
    controller: ParametricMPCController,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    E: np.ndarray,
    reference: np.ndarray,
    disturbance: np.ndarray,
    x0: np.ndarray,
    mpc_cfg: MPCConfig,
    search_cfg: SearchConfig,
    noise_std: float,
    seed: int,
) -> Dict[str, Any]:
    lower_bounds = np.array(
        [search_cfg.q_min_exp, search_cfg.q_min_exp, search_cfg.r_min_exp],
        dtype=float,
    )
    upper_bounds = np.array(
        [search_cfg.q_max_exp, search_cfg.q_max_exp, search_cfg.r_max_exp],
        dtype=float,
    )

    # ---------- global exploration ----------
    candidates = lhs_candidates(
        n_candidates=search_cfg.n_lhs_candidates,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        seed=seed,
    )

    best_theta: Optional[np.ndarray] = None
    best_cost = float("inf")

    for theta in candidates:
        cost = qr_objective(
            theta=theta,
            controller=controller,
            A=A,
            B=B,
            C=C,
            E=E,
            reference=reference,
            disturbance=disturbance,
            x0=x0,
            mpc_cfg=mpc_cfg,
            noise_std=noise_std,
            seed=seed,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        if cost < best_cost:
            best_cost = cost
            best_theta = np.asarray(theta, dtype=float)

    if best_theta is None or best_cost >= 1e6:
        return {
            "success": False,
            "reason": "No feasible candidate found in LHS search."
        }

    # ---------- local refinement ----------
    search_method = "LHS"
    if search_cfg.use_powell:
        def obj(theta: np.ndarray) -> float:
            return qr_objective(
                theta=theta,
                controller=controller,
                A=A,
                B=B,
                C=C,
                E=E,
                reference=reference,
                disturbance=disturbance,
                x0=x0,
                mpc_cfg=mpc_cfg,
                noise_std=noise_std,
                seed=seed,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )

        bounds = list(zip(lower_bounds.tolist(), upper_bounds.tolist()))
        result = minimize(
            fun=obj,
            x0=best_theta,
            method="Powell",
            bounds=bounds,
            options={
                "maxiter": search_cfg.powell_maxiter,
                "disp": False,
            },
        )

        theta_powell = np.asarray(result.x, dtype=float)
        cost_powell = float(result.fun)

        if cost_powell < best_cost:
            best_theta = theta_powell
            best_cost = cost_powell
            search_method = "LHS+Powell"

    q_diag, r_diag = theta_to_qr(best_theta)
    rng = np.random.default_rng(seed)

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
        rng=rng,
    )

    if not sim["feasible"]:
        return {
            "success": False,
            "reason": "Best candidate became infeasible on re-simulation."
        }

    metrics = compute_metrics(sim["y"], sim["u"], reference, mpc_cfg.Ts)

    return {
        "success": True,
        "q_diag": q_diag.tolist(),
        "r_diag": r_diag.tolist(),
        "Q": np.diag(q_diag),
        "R": np.diag(r_diag),
        "sim": sim,
        "metrics": metrics,
        "meta_cost": float(meta_cost(metrics)),
        "best_theta": best_theta.tolist(),
        "search_method": search_method,
    }


# ============================================================
# Streaming writer
# ============================================================

class DatasetWriter:
    def __init__(self, output_dir: Path, save_closed_loop_trajectories: bool, save_signals_npz: bool) -> None:
        self.output_dir = output_dir
        self.samples_dir = output_dir / "samples"
        self.samples_dir.mkdir(parents=True, exist_ok=True)

        self.save_closed_loop_trajectories = save_closed_loop_trajectories
        self.save_signals_npz = save_signals_npz

        self.summary_csv = output_dir / "summary.csv"
        self.failures_csv = output_dir / "failures.csv"
        self.manifest_jsonl = output_dir / "manifest.jsonl"

        self._init_csvs()

    def _init_csvs(self) -> None:
        if not self.summary_csv.exists():
            with self.summary_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "sample_id", "model_idx", "scenario_idx", "success",
                    "Ra", "La", "J", "b", "Kt", "Ke",
                    "gain_proxy", "Te", "Tm",
                    "reference_type", "ref_amplitude", "ref_secondary_amplitude",
                    "disturbance_amplitude", "noise_std",
                    "q1", "q2", "r1",
                    "iae", "ise", "control_energy", "control_variation",
                    "overshoot", "settling_time", "meta_cost",
                    "search_method", "sample_dir"
                ])
                writer.writeheader()

        if not self.failures_csv.exists():
            with self.failures_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "sample_id", "model_idx", "scenario_idx", "success", "reason"
                ])
                writer.writeheader()

    def write_success(self, record: Dict[str, Any], summary_row: Dict[str, Any], write_manifest: bool = False) -> None:
        sample_id = int(record["sample_id"])
        sample_dir = self.samples_dir / f"sample_{sample_id:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # meta.json
        meta_path = sample_dir / "meta.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        # optional signals.npz
        if self.save_signals_npz:
            np.savez_compressed(
                sample_dir / "signals.npz",
                reference=np.array(record["signals"]["reference"], dtype=float),
                disturbance=np.array(record["signals"]["disturbance"], dtype=float),
            )

        # optional closed-loop trajectories.npz
        if self.save_closed_loop_trajectories and "closed_loop" in record:
            np.savez_compressed(
                sample_dir / "closed_loop.npz",
                x_traj=np.array(record["closed_loop"]["x_traj"], dtype=float),
                y_traj=np.array(record["closed_loop"]["y_traj"], dtype=float),
                y_meas_traj=np.array(record["closed_loop"]["y_meas_traj"], dtype=float),
                u_traj=np.array(record["closed_loop"]["u_traj"], dtype=float),
                e_traj=np.array(record["closed_loop"]["e_traj"], dtype=float),
            )

        summary_row = dict(summary_row)
        summary_row["sample_dir"] = str(sample_dir)
        with self.summary_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
            writer.writerow(summary_row)

        if write_manifest:
            manifest_item = {
                "sample_id": sample_id,
                "sample_dir": str(sample_dir),
                "meta_path": str(meta_path),
            }
            with self.manifest_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(manifest_item, ensure_ascii=False) + "\n")

    def write_failure(self, failure_row: Dict[str, Any]) -> None:
        with self.failures_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(failure_row.keys()))
            writer.writerow(failure_row)

    def sample_done(self, sample_id: int) -> bool:
        sample_dir = self.samples_dir / f"sample_{sample_id:06d}"
        return (
            (sample_dir / "meta.json").exists()
            and (sample_dir / "signals.npz").exists()
            and (
                not self.save_closed_loop_trajectories
                or (sample_dir / "closed_loop.npz").exists()
            )
        )


# ============================================================
# Dataset generation (streaming)
# ============================================================

def generate_dataset_streaming(
    nominal_motor: MotorParams,
    mpc_cfg: MPCConfig,
    search_cfg: SearchConfig,
    dataset_cfg: DatasetConfig,
) -> Tuple[int, int, Optional[Dict[str, Any]]]:
    rng = np.random.default_rng(dataset_cfg.random_seed)

    output_dir = Path(dataset_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = DatasetWriter(
        output_dir=output_dir,
        save_closed_loop_trajectories=dataset_cfg.save_closed_loop_trajectories,
        save_signals_npz=dataset_cfg.save_signals_npz,
    )

    sample_id = 0
    n_success = 0
    n_total = dataset_cfg.n_models * dataset_cfg.scenarios_per_model
    first_success_record: Optional[Dict[str, Any]] = None

    t_global_start = time.perf_counter()

    for model_idx in range(dataset_cfg.n_models):
        motor_params = sample_motor_params(nominal_motor, rng)

        A_c, B_c, C_c, D_c = dc_motor_state_space(motor_params)
        A_d, B_d, C_d, D_d = discretize_system(A_c, B_c, C_c, D_c, mpc_cfg.Ts)
        E_d = disturbance_input_matrix(motor_params, mpc_cfg.Ts)
        tf_features = estimate_second_order_tf_like_features(motor_params)

        # Build controller once per model and reuse for all scenarios/candidates
        controller = ParametricMPCController(
            A=A_d,
            B=B_d,
            C=C_d,
            E=E_d,
            mpc_cfg=mpc_cfg,
        )

        for scenario_idx in range(dataset_cfg.scenarios_per_model):
            t0 = time.perf_counter()

            scfg = sample_scenario(mpc_cfg.sim_steps, rng)
            reference = generate_reference(scfg, mpc_cfg.sim_steps, mpc_cfg.Ts)
            disturbance = generate_disturbance(scfg, mpc_cfg.sim_steps)
            x0 = np.array(scfg.x0, dtype=float)

            seed = int(rng.integers(0, 1_000_000))

            if dataset_cfg.resume and writer.sample_done(sample_id):
                n_success += 1
                if first_success_record is None:
                    meta_path = output_dir / "samples" / f"sample_{sample_id:06d}" / "meta.json"
                    try:
                        with meta_path.open("r", encoding="utf-8") as f:
                            first_success_record = json.load(f)
                    except Exception:
                        first_success_record = None
                if dataset_cfg.print_every_sample:
                    print(
                        f"[SKIP] sample={sample_id:06d} "
                        f"model={model_idx} scenario={scenario_idx} already exists",
                        flush=True,
                    )
                sample_id += 1
                continue

            result = find_optimal_qr_fast(
                controller=controller,
                A=A_d,
                B=B_d,
                C=C_d,
                E=E_d,
                reference=reference,
                disturbance=disturbance,
                x0=x0,
                mpc_cfg=mpc_cfg,
                search_cfg=search_cfg,
                noise_std=scfg.measurement_noise_std,
                seed=seed,
            )

            elapsed = time.perf_counter() - t0

            if not result["success"]:
                failure_row = {
                    "sample_id": sample_id,
                    "model_idx": model_idx,
                    "scenario_idx": scenario_idx,
                    "success": False,
                    "reason": result["reason"],
                }
                writer.write_failure(failure_row)

                if dataset_cfg.print_every_sample:
                    print(
                        f"[FAIL] sample={sample_id:06d} "
                        f"model={model_idx} scenario={scenario_idx} "
                        f"time={elapsed:.2f}s reason={result['reason']}",
                        flush=True,
                    )

                sample_id += 1
                continue

            sim = result["sim"]
            metrics = result["metrics"]

            record = {
                "sample_id": sample_id,
                "model_idx": model_idx,
                "scenario_idx": scenario_idx,
                "success": True,
                "model": {
                    "A_d": A_d.tolist(),
                    "B_d": B_d.tolist(),
                    "C_d": C_d.tolist(),
                    "D_d": D_d.tolist(),
                    "E_d": E_d.tolist(),
                    "Ts": mpc_cfg.Ts,
                    "motor_params": asdict(motor_params),
                    "tf_features": tf_features,
                },
                "scenario": asdict(scfg),
                "constraints": {
                    "u_min": mpc_cfg.u_min,
                    "u_max": mpc_cfg.u_max,
                    "du_min": mpc_cfg.du_min,
                    "du_max": mpc_cfg.du_max,
                    "horizon": mpc_cfg.horizon,
                    "sim_steps": mpc_cfg.sim_steps,
                },
                "signals": {
                    "reference": reference.tolist(),
                    "disturbance": disturbance.tolist(),
                },
                "target": {
                    "Q_raw": result["Q"].tolist(),
                    "R_raw": result["R"].tolist(),
                    "Q_diag": result["q_diag"],
                    "R_diag": result["r_diag"],
                },
                "metrics": {
                    **metrics,
                    "meta_cost": result["meta_cost"],
                    "solver_status": sim["solver_status"],
                },
                "optimization": {
                    "search_method": result["search_method"],
                    "best_theta": result["best_theta"],
                    "elapsed_sec": elapsed,
                },
            }

            if dataset_cfg.save_closed_loop_trajectories:
                record["closed_loop"] = {
                    "x_traj": sim["x"].tolist(),
                    "y_traj": sim["y"].tolist(),
                    "y_meas_traj": sim["y_meas"].tolist(),
                    "u_traj": sim["u"].tolist(),
                    "e_traj": sim["e"].tolist(),
                }

            summary_row = {
                "sample_id": sample_id,
                "model_idx": model_idx,
                "scenario_idx": scenario_idx,
                "success": True,
                "Ra": motor_params.Ra,
                "La": motor_params.La,
                "J": motor_params.J,
                "b": motor_params.b,
                "Kt": motor_params.Kt,
                "Ke": motor_params.Ke,
                "gain_proxy": tf_features["gain_proxy"],
                "Te": tf_features["Te"],
                "Tm": tf_features["Tm"],
                "reference_type": scfg.reference_type,
                "ref_amplitude": scfg.ref_amplitude,
                "ref_secondary_amplitude": scfg.ref_secondary_amplitude,
                "disturbance_amplitude": scfg.disturbance_amplitude,
                "noise_std": scfg.measurement_noise_std,
                "q1": result["q_diag"][0],
                "q2": result["q_diag"][1],
                "r1": result["r_diag"][0],
                "iae": metrics["iae"],
                "ise": metrics["ise"],
                "control_energy": metrics["control_energy"],
                "control_variation": metrics["control_variation"],
                "overshoot": metrics["overshoot"],
                "settling_time": metrics["settling_time"],
                "meta_cost": result["meta_cost"],
                "search_method": result["search_method"],
                "sample_dir": "",
            }

            writer.write_success(
                record=record,
                summary_row=summary_row,
                write_manifest=dataset_cfg.chunk_jsonl_manifest,
            )

            if first_success_record is None:
                first_success_record = record

            n_success += 1

            if dataset_cfg.print_every_sample:
                print(
                    f"[OK]   sample={sample_id:06d} "
                    f"model={model_idx} scenario={scenario_idx} "
                    f"time={elapsed:.2f}s "
                    f"cost={result['meta_cost']:.4f} "
                    f"method={result['search_method']}",
                    flush=True,
                )

            sample_id += 1

    total_elapsed = time.perf_counter() - t_global_start
    print(
        f"\nFinished dataset generation: success={n_success}/{n_total}, "
        f"elapsed={total_elapsed/60.0:.2f} min, "
        f"output_dir={output_dir}",
        flush=True,
    )

    return n_success, n_total, first_success_record


# ============================================================
# Visualization
# ============================================================

def plot_sample(record: Dict[str, Any]) -> None:
    if "closed_loop" not in record:
        raise ValueError("This record does not contain closed-loop trajectories.")

    ref = np.array(record["signals"]["reference"], dtype=float)
    d = np.array(record["signals"]["disturbance"], dtype=float)
    x = np.array(record["closed_loop"]["x_traj"], dtype=float)
    y = np.array(record["closed_loop"]["y_traj"], dtype=float)
    u = np.array(record["closed_loop"]["u_traj"], dtype=float)
    Ts = float(record["model"]["Ts"])

    t_y = np.arange(len(y)) * Ts
    t_x = np.arange(len(x)) * Ts
    t_u = np.arange(len(u)) * Ts

    plt.figure(figsize=(10, 4))
    plt.plot(t_y, ref[:len(y)], label="reference")
    plt.plot(t_y, y, label="output y")
    plt.xlabel("Time [s]")
    plt.ylabel("Speed")
    plt.title(f"Sample {record['sample_id']} - Reference vs Output")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(t_x, x[:, 0], label="current i")
    plt.plot(t_x, x[:, 1], label="speed omega")
    plt.xlabel("Time [s]")
    plt.ylabel("States")
    plt.title(f"Sample {record['sample_id']} - State trajectories")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(t_u, u, label="control u")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage")
    plt.title(f"Sample {record['sample_id']} - Control input")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(t_y, d[:len(y)], label="disturbance")
    plt.xlabel("Time [s]")
    plt.ylabel("Load torque")
    plt.title(f"Sample {record['sample_id']} - Disturbance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_dataset_distributions(summary_csv_path: Path) -> None:
    if not summary_csv_path.exists():
        print("summary.csv not found.")
        return

    summary_df = pd.read_csv(summary_csv_path)
    ok = summary_df[summary_df["success"] == True].copy()
    if ok.empty:
        print("No successful samples to visualize.")
        return

    for col in ["q1", "q2", "r1", "meta_cost", "overshoot", "settling_time"]:
        plt.figure(figsize=(8, 4))
        plt.hist(ok[col].values, bins=30)
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(f"Distribution of {col}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a streaming QR dataset for the DC motor MPC problem.")
    parser.add_argument("--n_models", type=int, default=20)
    parser.add_argument("--scenarios_per_model", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="mpc_qr_dataset_streaming")
    parser.add_argument("--Ts", type=float, default=0.02)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--sim_steps", type=int, default=120)
    parser.add_argument("--u_min", type=float, default=-24.0)
    parser.add_argument("--u_max", type=float, default=24.0)
    parser.add_argument("--du_min", type=float, default=-4.0)
    parser.add_argument("--du_max", type=float, default=4.0)
    parser.add_argument("--solver_max_iter", type=int, default=4000)
    parser.add_argument("--n_lhs_candidates", type=int, default=12)
    parser.add_argument("--powell_maxiter", type=int, default=25)
    parser.add_argument("--q_min_exp", type=float, default=-3.0)
    parser.add_argument("--q_max_exp", type=float, default=3.0)
    parser.add_argument("--r_min_exp", type=float, default=-4.0)
    parser.add_argument("--r_max_exp", type=float, default=2.0)
    parser.add_argument("--Ra", type=float, default=2.0)
    parser.add_argument("--La", type=float, default=0.5)
    parser.add_argument("--J", type=float, default=0.02)
    parser.add_argument("--b", type=float, default=0.2)
    parser.add_argument("--Kt", type=float, default=0.1)
    parser.add_argument("--Ke", type=float, default=0.1)
    parser.add_argument("--params_json", type=str, default=None,
                        help="JSON produced by datasets/fit_motor_params_from_mat.py")
    parser.add_argument("--resume", action="store_true",
                        help="Skip complete sample folders that already exist in output_dir.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.params_json is not None:
        with open(args.params_json, "r", encoding="utf-8") as f:
            fitted = json.load(f)["params"]
        args.Ra = fitted["Ra"]
        args.La = fitted["La"]
        args.J = fitted["J"]
        args.b = fitted["b"]
        args.Kt = fitted["Kt"]
        args.Ke = fitted["Ke"]

    nominal_motor = MotorParams(
        Ra=args.Ra,
        La=args.La,
        J=args.J,
        b=args.b,
        Kt=args.Kt,
        Ke=args.Ke,
    )

    mpc_cfg = MPCConfig(
        Ts=args.Ts,
        horizon=args.horizon,
        sim_steps=args.sim_steps,
        u_min=args.u_min,
        u_max=args.u_max,
        du_min=args.du_min,
        du_max=args.du_max,
        solver="OSQP",
        solver_max_iter=args.solver_max_iter,
    )

    search_cfg = SearchConfig(
        n_lhs_candidates=args.n_lhs_candidates,
        use_powell=True,       # local refinement
        powell_maxiter=args.powell_maxiter,
        q_min_exp=args.q_min_exp,
        q_max_exp=args.q_max_exp,
        r_min_exp=args.r_min_exp,
        r_max_exp=args.r_max_exp,
    )

    dataset_cfg = DatasetConfig(
        n_models=args.n_models,
        scenarios_per_model=args.scenarios_per_model,
        random_seed=args.random_seed,
        output_dir=args.output_dir,
        save_closed_loop_trajectories=True,
        save_signals_npz=True,
        print_every_sample=not args.quiet,
        plot_first_successful_sample=False,
        chunk_jsonl_manifest=False,
        resume=args.resume,
    )

    n_success, n_total, first_success_record = generate_dataset_streaming(
        nominal_motor=nominal_motor,
        mpc_cfg=mpc_cfg,
        search_cfg=search_cfg,
        dataset_cfg=dataset_cfg,
    )

    print(f"Successful samples: {n_success} / {n_total}", flush=True)
    print(f"Summary CSV: {Path(dataset_cfg.output_dir) / 'summary.csv'}", flush=True)
    print(f"Failures CSV: {Path(dataset_cfg.output_dir) / 'failures.csv'}", flush=True)

    if dataset_cfg.plot_first_successful_sample and first_success_record is not None:
        plot_sample(first_success_record)
        plot_dataset_distributions(Path(dataset_cfg.output_dir) / "summary.csv")


if __name__ == "__main__":
    main()
