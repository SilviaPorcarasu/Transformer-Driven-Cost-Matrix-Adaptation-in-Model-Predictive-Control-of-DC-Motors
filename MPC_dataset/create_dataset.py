import os
import math
import json
import numpy as np

from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional

import control  # python-control
from scipy.linalg import expm


# -------------------------
# Utilities
# -------------------------

def set_seed(seed: int):
    np.random.seed(seed)

def random_orthonormal(n: int) -> np.ndarray:
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    return Q

def symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)

def sample_psd(n: int, scale=(1e-2, 1e2), jitter=1e-9) -> np.ndarray:
    """
    Random PSD-ish symmetric matrix. We enforce symmetry explicitly and
    add a tiny diagonal jitter to avoid numerical edge cases.
    """
    U, _ = np.linalg.qr(np.random.randn(n, n))
    eigs = 10 ** np.random.uniform(np.log10(scale[0]), np.log10(scale[1]), size=n)
    M = U @ np.diag(eigs) @ U.T
    M = symmetrize(M)
    M = M + jitter * np.eye(n)
    return M


def controllability_rank(A: np.ndarray, B: np.ndarray) -> int:
    n = A.shape[0]
    ctrb = B
    AB = B
    for _ in range(1, n):
        AB = A @ AB
        ctrb = np.concatenate([ctrb, AB], axis=1)
    return np.linalg.matrix_rank(ctrb)

def discretize_zoh(Ac: np.ndarray, Bc: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact ZOH discretization via matrix exponential.
    """
    n = Ac.shape[0]
    m = Bc.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = Ac
    M[:n, n:] = Bc
    Md = expm(M * dt)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    return Ad, Bd


def safe_dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
    """
    Discrete-time LQR with symmetry enforcement.
    """
    Qs = symmetrize(Q)
    Rs = symmetrize(R)

    # Optional: ensure R is positive definite enough for the solver
    # (dlqr typically expects R > 0)
    eps = 1e-9
    Rs = Rs + eps * np.eye(Rs.shape[0])

    K, S, E = control.dlqr(A, B, Qs, Rs)
    return np.array(K), np.array(S), np.array(E)


def clip_u(u: np.ndarray, u_max: Optional[float]) -> np.ndarray:
    if u_max is None:
        return u
    return np.clip(u, -u_max, u_max)


# -------------------------
# Disturbance generators
# -------------------------

def gen_disturbance(T: int, n: int, kind: str, dt: float) -> Tuple[np.ndarray, Dict]:
    """
    Returns d[t] shape (T, n): additive disturbance on state update.
    """
    meta = {"kind": kind}

    if kind == "none":
        return np.zeros((T, n)), meta

    if kind == "step":
        t0 = np.random.randint(T // 10, 9 * T // 10)
        amp = np.random.randn(n) * np.random.uniform(0.1, 2.0)
        d = np.zeros((T, n))
        d[t0:] = amp
        meta.update({"t0": int(t0), "amp": amp.tolist()})
        return d, meta

    if kind == "sine":
        freq = np.random.uniform(0.05, 2.0)  # Hz-ish
        amp = np.random.randn(n) * np.random.uniform(0.05, 1.0)
        phase = np.random.uniform(0, 2 * np.pi, size=n)
        t = np.arange(T) * dt
        d = np.sin(2 * np.pi * freq * t[:, None] + phase[None, :]) * amp[None, :]
        meta.update({"freq": float(freq), "amp": amp.tolist(), "phase": phase.tolist()})
        return d, meta

    if kind == "drift":
        drift = np.random.randn(n) * np.random.uniform(1e-4, 5e-2)
        d = np.cumsum(np.tile(drift[None, :], (T, 1)), axis=0)
        meta.update({"drift_per_step": drift.tolist()})
        return d, meta

    if kind == "impulse":
        k = np.random.randint(1, 6)
        d = np.zeros((T, n))
        idx = np.random.choice(np.arange(T), size=k, replace=False)
        for t0 in idx:
            d[t0] += np.random.randn(n) * np.random.uniform(0.5, 3.0)
        meta.update({"impulses": [int(i) for i in idx]})
        return d, meta

    raise ValueError(f"Unknown disturbance kind: {kind}")


# -------------------------
# System sampling
# -------------------------

@dataclass
class SystemSpec:
    n: int
    m: int
    dt: float
    A: np.ndarray
    B: np.ndarray
    stable_flag: bool  # whether sampled continuous-time A was stable-ish

def sample_continuous_system(n: int, m: int) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Sample (Ac, Bc). Some are unstable. Enforce controllability (approx) after discretization later.
    """
    # Mix stable/unstable eigenvalues by choosing real parts from a range that crosses 0
    stable_flag = (np.random.rand() < 0.5)

    # Create Ac with a desired spectrum-ish
    # Build via similarity transform: Ac = V diag(lambdas) V^{-1}
    V = np.random.randn(n, n)
    while np.linalg.matrix_rank(V) < n:
        V = np.random.randn(n, n)

    if stable_flag:
        real_parts = -10 ** np.random.uniform(-1, 1, size=n)  # ~[-10, -0.1]
    else:
        # some unstable modes
        real_parts = -10 ** np.random.uniform(-1, 1, size=n)
        k_unstable = np.random.randint(1, max(2, n // 2))
        idx = np.random.choice(np.arange(n), size=k_unstable, replace=False)
        real_parts[idx] = 10 ** np.random.uniform(-2, 0.7, size=k_unstable)  # ~[0.01, 5]
    lambdas = real_parts  # keep it real for simplicity

    Ac = V @ np.diag(lambdas) @ np.linalg.inv(V)

    # Control matrix
    Bc = np.random.randn(n, m)
    return Ac, Bc, stable_flag

def sample_discrete_system(n: int, m: int, dt_range=(0.01, 0.1), max_tries=200) -> SystemSpec:
    for _ in range(max_tries):
        dt = float(np.random.uniform(*dt_range))
        Ac, Bc, stable_flag = sample_continuous_system(n, m)
        A, B = discretize_zoh(Ac, Bc, dt)

        # Enforce controllability in discrete-time (stronger than stabilizable; good for synthetic data)
        if controllability_rank(A, B) == n:
            return SystemSpec(n=n, m=m, dt=dt, A=A, B=B, stable_flag=stable_flag)

    raise RuntimeError("Failed to sample a controllable discrete system.")


# -------------------------
# Episode simulation
# -------------------------

@dataclass
class EpisodeMeta:
    sys_id: int
    episode_id: int
    n_actual: int      # actual state dimension of this system
    m_actual: int      # actual input dimension of this system
    T: int
    disturbance: Dict
    u_max: Optional[float]
    process_noise_scale: float
    meas_noise_scale: float
    Q0_scale: Tuple[float, float]
    R0_scale: Tuple[float, float]
    eig_cl: list
    cost: float
    diverged: bool

def simulate_episode(
    sys: SystemSpec,
    T: int,
    x0_scale: float = 1.0,
    disturbance_kind: str = "step",
    process_noise_scale: float = 0.01,
    meas_noise_scale: float = 0.0,
    u_max: Optional[float] = None,
    Q0_scale=(1e-2, 1e2),
    R0_scale=(1e-2, 1e2),
    diverge_norm: float = 1e6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, EpisodeMeta]:
    """
    Simulate closed-loop with baseline (Q0,R0).
    Dynamics:
      x_{t+1} = A x_t + B u_t + d_t + w_t
    Optional measurement noise:
      y_t = x_t + v_t
      u_t computed from y_t
    """
    n, m = sys.n, sys.m
    A, B = sys.A, sys.B

    # ---- Robust LQR gain computation: resample Q0,R0 until dlqr succeeds ----
    max_lqr_tries = 30
    K = None
    Q0 = None
    R0 = None
    last_exc = None

    for _ in range(max_lqr_tries):
        Q0 = sample_psd(n, scale=Q0_scale)  # should be symmetric PSD
        R0 = sample_psd(m, scale=R0_scale)  # should be symmetric PSD (ideally PD)
        try:
            K, _, _ = safe_dlqr(A, B, Q0, R0)
            break
        except Exception as e:
            last_exc = e
            K = None

    # If still failing, return a "diverged" episode so caller can skip it
    if K is None:
        # Noise covariances (needed for shapes below)
        W = (process_noise_scale ** 2) * np.eye(n)
        V = (meas_noise_scale ** 2) * np.eye(n)
        d, dmeta = gen_disturbance(T, n, disturbance_kind, sys.dt)

        x = np.zeros((T + 1, n), dtype=float)
        u = np.zeros((T, m), dtype=float)
        y = np.zeros((T, n), dtype=float)

        meta = EpisodeMeta(
            sys_id=-1,  # set by caller
            episode_id=-1,
            n_actual=n,
            m_actual=m,
            T=T,
            disturbance={**dmeta, "note": "dlqr_failed", "error": str(last_exc)},
            u_max=u_max,
            process_noise_scale=process_noise_scale,
            meas_noise_scale=meas_noise_scale,
            Q0_scale=Q0_scale,
            R0_scale=R0_scale,
            eig_cl=[],
            cost=float("inf"),
            diverged=True,
        )
        return x, u, y, Q0, R0, meta
    # ------------------------------------------------------------------------

    Acl = A - B @ K
    eig_cl = np.linalg.eigvals(Acl)

    # Noise covariances
    W = (process_noise_scale ** 2) * np.eye(n)
    V = (meas_noise_scale ** 2) * np.eye(n)

    d, dmeta = gen_disturbance(T, n, disturbance_kind, sys.dt)

    x = np.zeros((T + 1, n))
    u = np.zeros((T, m))
    y = np.zeros((T, n))
    x[0] = np.random.randn(n) * x0_scale

    cost = 0.0
    diverged = False

    for t in range(T):
        v = np.random.multivariate_normal(np.zeros(n), V)
        y[t] = x[t] + v

        u_t = -K @ y[t]
        u_t = clip_u(u_t, u_max)
        u[t] = u_t

        w = np.random.multivariate_normal(np.zeros(n), W)
        x[t + 1] = A @ x[t] + B @ u_t + d[t] + w

        cost += float(x[t].T @ Q0 @ x[t] + u_t.T @ R0 @ u_t)

        if np.linalg.norm(x[t + 1]) > diverge_norm or not np.isfinite(x[t + 1]).all():
            diverged = True
            # truncate by keeping remaining zeros (caller may filter)
            break

    meta = EpisodeMeta(
        sys_id=-1,  # set by caller
        episode_id=-1,
        n_actual=n,
        m_actual=m,
        T=T,
        disturbance=dmeta,
        u_max=u_max,
        process_noise_scale=process_noise_scale,
        meas_noise_scale=meas_noise_scale,
        Q0_scale=Q0_scale,
        R0_scale=R0_scale,
        eig_cl=[float(abs(v)) for v in eig_cl],
        cost=cost,
        diverged=diverged,
    )
    return x, u, y, Q0, R0, meta



# -------------------------
# Padding utility
# -------------------------

def pad_array(arr, target_shape):
    """Zero-pad an array's last dimensions to target_shape."""
    pad_widths = []
    offset = arr.ndim - len(target_shape)
    for i in range(arr.ndim):
        if i < offset:
            pad_widths.append((0, 0))
        else:
            actual = arr.shape[i]
            target = target_shape[i - offset]
            pad_widths.append((0, target - actual))
    return np.pad(arr, pad_widths, mode='constant', constant_values=0.0)


# -------------------------
# Dataset generation (sharded)
# -------------------------

def generate_dataset(
    out_dir: str,
    num_systems: int = 200,
    episodes_per_system: int = 50,
    n_range: Tuple[int, int] = (2, 10),
    m_range: Tuple[int, int] = (1, 5),
    n_max: int = 10,
    m_max: int = 5,
    T: int = 256,
    seed: int = 0,
    shard_size: int = 500,
):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)

    disturbance_choices = ["none", "step", "sine", "drift", "impulse"]
    u_max_choices = [None, 0.5, 1.0, 2.0]
    proc_noise_choices = [0.0, 0.001, 0.01, 0.05]
    meas_noise_choices = [0.0, 0.001, 0.01]
    x0_scale_choices = [0.1, 0.5, 1.0, 2.0]

    all_records = []
    shard_idx = 0

    for sys_id in range(num_systems):
        # Sample random dimensions for this system
        n_i = np.random.randint(n_range[0], n_range[1] + 1)
        m_i = np.random.randint(m_range[0], min(m_range[1], n_i) + 1)  # m <= n

        sys = sample_discrete_system(n=n_i, m=m_i)
        for ep in range(episodes_per_system):
            disturbance_kind = np.random.choice(disturbance_choices)
            u_max = np.random.choice(u_max_choices)
            process_noise_scale = float(np.random.choice(proc_noise_choices))
            meas_noise_scale = float(np.random.choice(meas_noise_choices))
            x0_scale = float(np.random.choice(x0_scale_choices))

            x, u, y, Q0, R0, meta = simulate_episode(
                sys=sys,
                T=T,
                x0_scale=x0_scale,
                disturbance_kind=disturbance_kind,
                process_noise_scale=process_noise_scale,
                meas_noise_scale=meas_noise_scale,
                u_max=u_max,
            )
            meta.sys_id = sys_id
            meta.episode_id = ep

            # Filter out totally useless rollouts (optional)
            if meta.diverged:
                continue

            # Pad all arrays to (n_max, m_max) for uniform shard shapes
            record = {
                "A": pad_array(sys.A, (n_max, n_max)).astype(np.float32),
                "B": pad_array(sys.B, (n_max, m_max)).astype(np.float32),
                "dt": np.float32(sys.dt),
                "x": pad_array(x, (n_max,)).astype(np.float32),
                "u": pad_array(u, (m_max,)).astype(np.float32),
                "y": pad_array(y, (n_max,)).astype(np.float32),
                "Q0": pad_array(Q0, (n_max, n_max)).astype(np.float32),
                "R0": pad_array(R0, (m_max, m_max)).astype(np.float32),
                "meta": asdict(meta),
            }
            all_records.append(record)

            # shard flush
            if len(all_records) >= shard_size:
                shard_path = os.path.join(out_dir, f"synthetic_shard_{shard_idx:04d}.npz")
                save_shard_npz(shard_path, all_records)
                all_records = []
                shard_idx += 1

    # final shard
    if all_records:
        shard_path = os.path.join(out_dir, f"synthetic_shard_{shard_idx:04d}.npz")
        save_shard_npz(shard_path, all_records)

    # write manifest
    manifest = {
        "num_systems": num_systems,
        "episodes_per_system": episodes_per_system,
        "n_max": n_max, "m_max": m_max,
        "n_range": list(n_range), "m_range": list(m_range),
        "T": T,
        "seed": seed,
        "shard_size": shard_size,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

def save_shard_npz(path: str, records: list):
    """
    Save a list of episode dicts into one npz.
    We store arrays as object arrays for variable metadata.
    If you want max throughput, switch to HDF5/Zarr later.
    """
    A = np.stack([r["A"] for r in records], axis=0)
    B = np.stack([r["B"] for r in records], axis=0)
    dt = np.stack([r["dt"] for r in records], axis=0)
    x = np.stack([r["x"] for r in records], axis=0)
    u = np.stack([r["u"] for r in records], axis=0)
    y = np.stack([r["y"] for r in records], axis=0)
    Q0 = np.stack([r["Q0"] for r in records], axis=0)
    R0 = np.stack([r["R0"] for r in records], axis=0)
    meta = np.array([json.dumps(r["meta"]) for r in records], dtype=object)

    np.savez_compressed(path, A=A, B=B, dt=dt, x=x, u=u, y=y, Q0=Q0, R0=R0, meta=meta)


if __name__ == "__main__":
    generate_dataset(
        out_dir="synthetic_lqr_data",
        num_systems=200,
        episodes_per_system=50,
        n_range=(2, 10),
        m_range=(1, 5),
        n_max=10,
        m_max=5,
        T=256,
        seed=42,
        shard_size=500,
    )
    print("Done.")
