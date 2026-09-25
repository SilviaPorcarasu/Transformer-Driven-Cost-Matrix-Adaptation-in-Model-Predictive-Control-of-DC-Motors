from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np


def load_sample(sample_dir: str | Path) -> Dict[str, Any]:
    sample_dir = Path(sample_dir)

    meta_path = sample_dir / "meta.json"
    signals_path = sample_dir / "signals.npz"
    closed_loop_path = sample_dir / "closed_loop.npz"

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    signals = {}
    if signals_path.exists():
        npz = np.load(signals_path)
        signals = {k: npz[k] for k in npz.files}

    closed_loop = {}
    if closed_loop_path.exists():
        npz = np.load(closed_loop_path)
        closed_loop = {k: npz[k] for k in npz.files}

    return {
        "meta": meta,
        "signals_npz": signals,
        "closed_loop_npz": closed_loop,
    }


def print_sample_info(sample: Dict[str, Any]) -> None:
    meta = sample["meta"]

    print("\n" + "=" * 80)
    print(f"SAMPLE ID: {meta['sample_id']}")
    print(f"MODEL IDX: {meta['model_idx']}")
    print(f"SCENARIO IDX: {meta['scenario_idx']}")
    print(f"SUCCESS: {meta['success']}")

    print("\n--- MODEL ---")
    print("Ts:", meta["model"]["Ts"])
    print("A_d:")
    print(np.array(meta["model"]["A_d"]))
    print("B_d:")
    print(np.array(meta["model"]["B_d"]))
    print("C_d:")
    print(np.array(meta["model"]["C_d"]))
    print("D_d:")
    print(np.array(meta["model"]["D_d"]))
    print("E_d:")
    print(np.array(meta["model"]["E_d"]))

    print("\nMotor params:")
    for k, v in meta["model"]["motor_params"].items():
        print(f"  {k}: {v}")

    print("\nTF-like features:")
    for k, v in meta["model"]["tf_features"].items():
        print(f"  {k}: {v}")

    print("\n--- SCENARIO ---")
    for k, v in meta["scenario"].items():
        print(f"  {k}: {v}")

    print("\n--- CONSTRAINTS ---")
    for k, v in meta["constraints"].items():
        print(f"  {k}: {v}")

    print("\n--- TARGET ---")
    print("Q_raw:")
    print(np.array(meta["target"]["Q_raw"]))
    print("R_raw:")
    print(np.array(meta["target"]["R_raw"]))
    print("Q_diag:", meta["target"]["Q_diag"])
    print("R_diag:", meta["target"]["R_diag"])

    print("\n--- METRICS ---")
    for k, v in meta["metrics"].items():
        print(f"  {k}: {v}")

    if "optimization" in meta:
        print("\n--- OPTIMIZATION ---")
        for k, v in meta["optimization"].items():
            print(f"  {k}: {v}")

    print("=" * 80 + "\n")


def plot_sample_full(sample: Dict[str, Any]) -> None:
    meta = sample["meta"]

    Ts = float(meta["model"]["Ts"])

    if sample["signals_npz"]:
        reference = sample["signals_npz"].get("reference", None)
        disturbance = sample["signals_npz"].get("disturbance", None)
    else:
        reference = np.array(meta["signals"]["reference"], dtype=float)
        disturbance = np.array(meta["signals"]["disturbance"], dtype=float)

    if not sample["closed_loop_npz"]:
        raise ValueError("closed_loop.npz not found for this sample.")

    closed = sample["closed_loop_npz"]
    x_traj = closed["x_traj"]
    y_traj = closed["y_traj"]
    y_meas_traj = closed["y_meas_traj"]
    u_traj = closed["u_traj"]
    e_traj = closed["e_traj"]

    t_y = np.arange(len(y_traj)) * Ts
    t_x = np.arange(len(x_traj)) * Ts
    t_u = np.arange(len(u_traj)) * Ts

    plt.figure(figsize=(10, 4))
    plt.plot(t_y, reference[:len(y_traj)], label="reference")
    plt.plot(t_y, y_traj, label="output y")
    plt.plot(t_y, y_meas_traj, label="measured y", alpha=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Output")
    plt.title(f"Sample {meta['sample_id']} - Reference vs Output")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(t_x, x_traj[:, 0], label="current i")
    plt.plot(t_x, x_traj[:, 1], label="speed omega")
    plt.xlabel("Time [s]")
    plt.ylabel("States")
    plt.title(f"Sample {meta['sample_id']} - State trajectories")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(t_u, u_traj, label="control u")
    plt.xlabel("Time [s]")
    plt.ylabel("Control")
    plt.title(f"Sample {meta['sample_id']} - Control input")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(t_y, e_traj, label="tracking error")
    plt.xlabel("Time [s]")
    plt.ylabel("Error")
    plt.title(f"Sample {meta['sample_id']} - Tracking error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(t_y, disturbance[:len(y_traj)], label="disturbance")
    plt.xlabel("Time [s]")
    plt.ylabel("Disturbance")
    plt.title(f"Sample {meta['sample_id']} - Disturbance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def inspect_one_sample(sample_dir: str | Path) -> None:
    sample = load_sample(sample_dir)
    print_sample_info(sample)
    plot_sample_full(sample)

import random
from pathlib import Path


def inspect_random_samples(dataset_dir: str | Path, n_samples: int = 3, seed: int = 42) -> None:
    rng = random.Random(seed)
    dataset_dir = Path(dataset_dir)
    samples_dir = dataset_dir / "samples"

    sample_dirs = sorted([p for p in samples_dir.iterdir() if p.is_dir()])
    if not sample_dirs:
        print("No sample directories found.")
        return

    chosen = rng.sample(sample_dirs, min(n_samples, len(sample_dirs)))

    for sample_dir in chosen:
        inspect_one_sample(sample_dir)


inspect_random_samples("mpc_qr_dataset_streaming", n_samples=6)