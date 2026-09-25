from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import solve_discrete_are


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def _theta_to_qr(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.array(
        [
            float(row["new_theta_q1"]),
            float(row["new_theta_q2"]),
            float(row["new_theta_r"]),
        ],
        dtype=np.float64,
    )
    Q = np.diag([10.0 ** theta[0], 10.0 ** theta[1]]).astype(np.float64)
    R = np.array([[10.0 ** theta[2]]], dtype=np.float64)
    return theta, Q, R


def build(input_root: Path, recalibration_csv: Path, output_root: Path, overwrite: bool) -> None:
    if overwrite and output_root.exists():
        shutil.rmtree(output_root)
    samples_out = output_root / "samples"
    samples_out.mkdir(parents=True, exist_ok=True)

    rows = _read_csv(recalibration_csv)
    summary_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    for out_idx, row in enumerate(rows):
        sample_name = row["sample"]
        src_dir = input_root / "samples" / sample_name
        dst_dir = samples_out / f"sample_{out_idx:06d}"
        try:
            meta = _load_json(src_dir / "meta.json")
            theta, Q, R = _theta_to_qr(row)
            A = np.asarray(meta["model"]["A_d"], dtype=np.float64)
            B = np.asarray(meta["model"]["B_d"], dtype=np.float64)
            P = solve_discrete_are(A, B, Q, R)
            P = 0.5 * (P + P.T)

            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_dir / "signals.npz", dst_dir / "signals.npz")
            if (src_dir / "closed_loop.npz").exists():
                shutil.copy2(src_dir / "closed_loop.npz", dst_dir / "closed_loop.npz")

            meta["sample_id"] = out_idx
            meta["source_sample"] = sample_name
            meta["target"] = {
                "Q_raw": Q.astype(float).tolist(),
                "R_raw": R.astype(float).tolist(),
                "Q_diag": np.diag(Q).astype(float).tolist(),
                "R_diag": np.diag(R).astype(float).tolist(),
                "P_raw": P.astype(float).tolist(),
                "P_diag": np.diag(P).astype(float).tolist(),
            }
            meta["metrics"] = {
                **meta.get("metrics", {}),
                "spec_cost_old": float(row.get("old_spec_cost", np.nan)),
                "spec_cost_new": float(row.get("new_spec_cost", np.nan)),
                "spec_improvement_ratio": float(row.get("improvement_ratio", np.nan)),
                "spec_iae": float(row.get("iae", np.nan)),
                "spec_ise": float(row.get("ise", np.nan)),
                "spec_control_energy": float(row.get("control_energy", np.nan)),
                "spec_control_variation": float(row.get("control_variation", np.nan)),
                "spec_overshoot": float(row.get("overshoot", np.nan)),
                "spec_settling_time": float(row.get("settling_time", np.nan)),
            }
            meta["optimization"] = {
                **meta.get("optimization", {}),
                "search_method": "spec_based_recalibration",
                "best_theta": theta.astype(float).tolist(),
                "source_recalibration_csv": str(recalibration_csv),
            }
            _write_json(dst_dir / "meta.json", meta)

            summary_rows.append(
                {
                    "sample_id": out_idx,
                    "source_sample": sample_name,
                    "q1": float(Q[0, 0]),
                    "q2": float(Q[1, 1]),
                    "r1": float(R[0, 0]),
                    "theta_q1": float(theta[0]),
                    "theta_q2": float(theta[1]),
                    "theta_r": float(theta[2]),
                    "p_trace": float(np.trace(P)),
                    "spec_cost_old": float(row.get("old_spec_cost", np.nan)),
                    "spec_cost_new": float(row.get("new_spec_cost", np.nan)),
                    "spec_improvement_ratio": float(row.get("improvement_ratio", np.nan)),
                }
            )
        except Exception as exc:
            failed_rows.append({"source_sample": sample_name, "error": repr(exc)})

    _write_csv(output_root / "summary.csv", summary_rows)
    _write_csv(output_root / "failures.csv", failed_rows)
    manifest = {
        "input_root": str(input_root),
        "recalibration_csv": str(recalibration_csv),
        "output_root": str(output_root),
        "converted": len(summary_rows),
        "failed": len(failed_rows),
        "format": "motor_pqr_sample_folder",
        "target_source": "spec_based_recalibration",
    }
    _write_json(output_root / "recalibrated_manifest.json", manifest)
    print(f"Done. converted={len(summary_rows)} failed={len(failed_rows)} output={output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build trainable PQR dataset from spec recalibration CSV.")
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--recalibration_csv", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build(Path(args.input_root), Path(args.recalibration_csv), Path(args.output_root), args.overwrite)


if __name__ == "__main__":
    main()
