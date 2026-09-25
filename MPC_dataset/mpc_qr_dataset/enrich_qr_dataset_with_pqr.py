from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
from scipy.linalg import solve_discrete_are


DEFAULT_INPUT_ROOT = Path("MPC_dataset/mpc_qr_dataset/mpc_qr_dataset_streaming")
DEFAULT_OUTPUT_ROOT = Path("MPC_dataset/mpc_qr_dataset/mpc_pqr_dataset_streaming")


def _as_array(value: Any, dtype=np.float64) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _steady_state_reference(A: np.ndarray, B: np.ndarray, C: np.ndarray, refs: np.ndarray) -> np.ndarray:
    """Return x_ref[:, k] from the least-squares steady-state equations."""
    n = A.shape[0]
    m = B.shape[1]
    lhs = np.block([
        [np.eye(n) - A, -B],
        [C, np.zeros((C.shape[0], m))],
    ])

    x_ref = np.zeros((n, refs.shape[0]), dtype=np.float32)
    for k, ref in enumerate(refs):
        rhs = np.concatenate([np.zeros(n), np.array([float(ref)])])
        sol, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        x_ref[:, k] = sol[:n].astype(np.float32)
    return x_ref


def _reachable_output_proxy(A: np.ndarray, B: np.ndarray, C: np.ndarray, u_max: float) -> float:
    try:
        dc_gain = C @ np.linalg.inv(np.eye(A.shape[0]) - A) @ B
        return float(abs(dc_gain.item()) * abs(u_max))
    except np.linalg.LinAlgError:
        return float(abs(u_max))


def _build_tokens_static(meta: Dict[str, Any], A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    params = meta["model"]["motor_params"]
    tf = meta["model"].get("tf_features", {})
    scenario = meta["scenario"]
    constraints = meta["constraints"]

    x0 = np.asarray(scenario.get("x0", [0.0, 0.0]), dtype=np.float32)
    if x0.shape[0] < 2:
        x0 = np.pad(x0, (0, 2 - x0.shape[0]))

    reachable = _reachable_output_proxy(A, B, C, float(constraints.get("u_max", 24.0)))

    return np.array([
        params.get("Ra", 0.0),
        params.get("La", 0.0),
        params.get("J", 0.0),
        params.get("b", 0.0),
        params.get("Kt", 0.0),
        params.get("Ke", 0.0),
        tf.get("gain_proxy", 0.0),
        tf.get("Te", 0.0),
        tf.get("Tm", 0.0),
        x0[0],
        x0[1],
        scenario.get("ref_amplitude", 0.0),
        scenario.get("ref_secondary_amplitude", 0.0),
        scenario.get("disturbance_amplitude", 0.0),
        scenario.get("measurement_noise_std", 0.0),
        reachable,
    ], dtype=np.float32)


def _build_tokens_seq(reference: np.ndarray, disturbance: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
    T = reference.shape[0]
    time_norm = np.linspace(0.0, 1.0, T, dtype=np.float32)

    # Keep the historical 5-column PQR-debug layout. Column 3 duplicates reference
    # and can be dropped by the existing dataset loader.
    return np.column_stack([
        reference.astype(np.float32),
        disturbance.astype(np.float32),
        x_ref[0].astype(np.float32),
        reference.astype(np.float32),
        time_norm,
    ]).astype(np.float32)


def _iter_sample_dirs(root: Path) -> Iterable[Path]:
    samples_dir = root / "samples"
    if not samples_dir.exists() and (root / "samples_1").exists():
        samples_dir = root / "samples_1"
    if not samples_dir.exists():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")
    yield from sorted(samples_dir.glob("sample_*"))


def enrich_sample(input_dir: Path, output_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    meta_path = input_dir / "meta.json"
    signals_path = input_dir / "signals.npz"
    closed_loop_path = input_dir / "closed_loop.npz"

    if not meta_path.exists() or not signals_path.exists():
        return False, {"reason": "missing meta.json or signals.npz"}

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    if not meta.get("success", False):
        return False, {"reason": "sample not successful"}

    model = meta["model"]
    target = meta["target"]
    A = _as_array(model["A_d"])
    B = _as_array(model["B_d"])
    C = _as_array(model["C_d"])
    Q = _as_array(target["Q_raw"])
    R = _as_array(target["R_raw"])

    P = solve_discrete_are(A, B, Q, R)
    P = 0.5 * (P + P.T)

    signals = np.load(signals_path)
    reference = np.asarray(signals["reference"], dtype=np.float32)
    disturbance = np.asarray(signals["disturbance"], dtype=np.float32)
    if reference.shape != disturbance.shape:
        raise ValueError(f"{input_dir.name}: reference/disturbance shape mismatch")

    x_ref = _steady_state_reference(A, B, C, reference)
    tokens_seq = _build_tokens_seq(reference, disturbance, x_ref)
    tokens_static = _build_tokens_static(meta, A, B, C)

    output_dir.mkdir(parents=True, exist_ok=True)

    target["P_raw"] = P.astype(float).tolist()
    target["P_diag"] = np.diag(P).astype(float).tolist()

    meta["signals"]["x_ref"] = x_ref.astype(float).tolist()
    meta["signals"]["tokens_seq"] = tokens_seq.astype(float).tolist()
    meta["signals"]["tokens_static"] = tokens_static.astype(float).tolist()

    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    np.savez_compressed(
        output_dir / "signals.npz",
        reference=reference,
        disturbance=disturbance,
        x_ref=x_ref,
        tokens_seq=tokens_seq,
        tokens_static=tokens_static,
    )

    if closed_loop_path.exists():
        shutil.copy2(closed_loop_path, output_dir / "closed_loop.npz")

    row = {
        "sample_id": meta.get("sample_id", input_dir.name),
        "success": True,
        "p1": float(P[0, 0]),
        "p2": float(P[1, 1]),
        "q1": float(Q[0, 0]),
        "q2": float(Q[1, 1]),
        "r1": float(R[0, 0]),
        "tokens_seq_shape": str(tuple(tokens_seq.shape)),
        "tokens_static_shape": str(tuple(tokens_static.shape)),
        "sample_dir": str(output_dir),
    }
    return True, row


def write_summary(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the DC motor QR dataset into a PQR-compatible dataset."
    )
    parser.add_argument("--input_root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    input_root = args.input_root
    output_root = args.output_root

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_samples = output_root / "samples"
    output_samples.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = 0
    failed = 0

    for idx, sample_dir in enumerate(_iter_sample_dirs(input_root)):
        if args.limit is not None and idx >= args.limit:
            break
        out_dir = output_samples / sample_dir.name
        try:
            ok, info = enrich_sample(sample_dir, out_dir)
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {sample_dir.name}: {exc}")
            continue
        if ok:
            rows.append(info)
        else:
            skipped += 1
            print(f"[SKIP] {sample_dir.name}: {info.get('reason', 'unknown')}")

    write_summary(output_root / "summary.csv", rows)
    print(
        f"Done. converted={len(rows)} skipped={skipped} failed={failed} "
        f"output={output_root}"
    )


if __name__ == "__main__":
    main()
