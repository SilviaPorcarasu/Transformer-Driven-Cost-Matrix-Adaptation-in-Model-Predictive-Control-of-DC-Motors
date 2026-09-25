from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import MPCDataset, get_manifest
from model import HybridControllerModel
from train import dare_iterate, frob_rel, riccati_residual, split_indices


def _to_float(x):
    return float(np.asarray(x).item())


def _stats(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    p25, p50, p75, p95 = np.percentile(values, [25, 50, 75, 95])
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p25": float(p25),
        "median": float(p50),
        "p75": float(p75),
        "p95": float(p95),
        "max": float(np.max(values)),
    }


def _frob_rel_per_sample(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    num = torch.linalg.norm(pred - target, ord="fro", dim=(-2, -1))
    den = torch.linalg.norm(target, ord="fro", dim=(-2, -1)) + eps
    return num / den


def _mae_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target), dim=(-2, -1))


def _rmse_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2, dim=(-2, -1)))


def _global_r2(pred: np.ndarray, target: np.ndarray) -> float:
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    ss_res = np.sum((target_flat - pred_flat) ** 2)
    ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _safe_log10_diag(matrix: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    diag = torch.diagonal(matrix, dim1=-2, dim2=-1).clamp_min(eps)
    return torch.log10(diag)


def _per_sample_r2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    ss_res = ((target_flat - pred_flat) ** 2).sum(dim=-1)
    ss_tot = ((target_flat - target_flat.mean(dim=-1, keepdim=True)) ** 2).sum(dim=-1)
    r2 = 1.0 - ss_res / ss_tot.clamp_min(1e-12)
    r2[ss_tot <= 1e-12] = float("nan")
    return r2


def _load_normalization(run_dir: Path):
    stats_path = run_dir / "normalization.npz"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing normalization file: {stats_path}")
    data = np.load(stats_path)
    return data["seq_mean"], data["seq_std"], data["static_mean"], data["static_std"]


def _load_target_normalization(run_dir: Path):
    stats_path = run_dir / "normalization.npz"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing normalization file: {stats_path}")
    data = np.load(stats_path)
    if "target_mean" not in data or "target_std" not in data:
        return None, None
    return (
        torch.tensor(data["target_mean"], dtype=torch.float32),
        torch.tensor(data["target_std"], dtype=torch.float32),
    )


def _load_config(checkpoint: Path) -> Dict:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {})
    config_path = checkpoint.parent / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        config.update(saved_args)
        return config
    return saved_args


def _denormalize_log_targets(
    pred_log: torch.Tensor,
    target_mean: torch.Tensor | None,
    target_std: torch.Tensor | None,
) -> torch.Tensor:
    if target_mean is None or target_std is None:
        return pred_log
    return pred_log * target_std.to(pred_log.device) + target_mean.to(pred_log.device)


@torch.no_grad()
def collect_predictions(
    model,
    loader,
    device: torch.device,
    target_mode: str = "full_psd",
    target_mean: torch.Tensor | None = None,
    target_std: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    out = {
        "P_pred": [], "Q_pred": [], "R_pred": [], "u_pred": [],
        "P0": [], "Q0": [], "R0": [], "u0": [],
        "A": [], "B": [],
    }
    model.eval()
    for batch in loader:
        tokens_seq = batch["tokens_seq"].to(device)
        tokens_static = batch["tokens_static"].to(device)
        pred = model(tokens_seq, tokens_static)
        if target_mode == "qr_log_diag":
            if len(pred) == 3:
                Qp, Rp, up = pred
            else:
                Qp, Rp = pred
                up = torch.zeros_like(batch["u"].to(device))
            n = batch["Q0"].shape[-1]
            m = batch["R0"].shape[-1]
            pred_log = torch.cat([Qp, Rp], dim=-1)
            pred_log = _denormalize_log_targets(pred_log, target_mean, target_std)
            Qp_log = pred_log[:, :n]
            Rp_log = pred_log[:, n:n + m]
            Qp = torch.diag_embed(torch.pow(10.0, Qp_log))
            Rp = torch.diag_embed(torch.pow(10.0, Rp_log))
            Pp = dare_iterate(batch["A"].to(device), batch["B"].to(device), Qp, Rp, n_iter=80)
        else:
            if len(pred) == 4:
                Pp, Qp, Rp, up = pred
            else:
                Pp, Qp, Rp = pred
                up = torch.zeros_like(batch["u"].to(device))

            if target_mode == "log_diag":
                n = batch["P0"].shape[-1]
                m = batch["R0"].shape[-1]
                pred_log = torch.cat([Pp, Qp, Rp], dim=-1)
                pred_log = _denormalize_log_targets(pred_log, target_mean, target_std)
                Pp_log = pred_log[:, :n]
                Qp_log = pred_log[:, n:n + n]
                Rp_log = pred_log[:, n + n:n + n + m]
                Pp = torch.diag_embed(torch.pow(10.0, Pp_log))
                Qp = torch.diag_embed(torch.pow(10.0, Qp_log))
                Rp = torch.diag_embed(torch.pow(10.0, Rp_log))

        out["P_pred"].append(Pp.cpu())
        out["Q_pred"].append(Qp.cpu())
        out["R_pred"].append(Rp.cpu())
        out["u_pred"].append(up.cpu())
        out["P0"].append(batch["P0"].cpu())
        out["Q0"].append(batch["Q0"].cpu())
        out["R0"].append(batch["R0"].cpu())
        out["u0"].append(batch["u"].cpu())
        out["A"].append(batch["A"].cpu())
        out["B"].append(batch["B"].cpu())
    return {key: torch.cat(value, dim=0) for key, value in out.items()}


def compute_metrics(results: Dict[str, torch.Tensor]) -> Dict:
    metrics = {}
    per_sample_rows = []

    for name in ["P", "Q", "R"]:
        pred = results[f"{name}_pred"]
        target = results[f"{name}0"]
        rel = _frob_rel_per_sample(pred, target).numpy()
        mae = _mae_per_sample(pred, target).numpy()
        rmse = _rmse_per_sample(pred, target).numpy()
        r2_sample = _per_sample_r2(pred, target).numpy()
        pred_log = _safe_log10_diag(pred)
        target_log = _safe_log10_diag(target)
        log_abs = torch.mean(torch.abs(pred_log - target_log), dim=-1).numpy()
        log_rmse = torch.sqrt(torch.mean((pred_log - target_log) ** 2, dim=-1)).numpy()
        metrics[name] = {
            "frob_rel": _stats(rel),
            "mae": _stats(mae),
            "rmse": _stats(rmse),
            "r2_per_sample": _stats(r2_sample),
            "r2_global": _global_r2(pred.numpy(), target.numpy()),
            "log10_diag_mae": _stats(log_abs),
            "log10_diag_rmse": _stats(log_rmse),
            "log10_diag_r2_global": _global_r2(pred_log.numpy(), target_log.numpy()),
        }

    ric = riccati_residual(
        results["P_pred"], results["Q_pred"], results["R_pred"], results["A"], results["B"]
    )
    ric_target = riccati_residual(
        results["P0"], results["Q0"], results["R0"], results["A"], results["B"]
    )
    u_rmse = torch.sqrt(torch.mean((results["u_pred"] - results["u0"]) ** 2, dim=(1, 2))).numpy()
    metrics["riccati_residual_pred_mean_batch"] = float(ric)
    metrics["riccati_residual_target_mean_batch"] = float(ric_target)
    metrics["u_rmse"] = _stats(u_rmse)

    rels = {
        "p_rel": _frob_rel_per_sample(results["P_pred"], results["P0"]).numpy(),
        "q_rel": _frob_rel_per_sample(results["Q_pred"], results["Q0"]).numpy(),
        "r_rel": _frob_rel_per_sample(results["R_pred"], results["R0"]).numpy(),
        "p_mae": _mae_per_sample(results["P_pred"], results["P0"]).numpy(),
        "q_mae": _mae_per_sample(results["Q_pred"], results["Q0"]).numpy(),
        "r_mae": _mae_per_sample(results["R_pred"], results["R0"]).numpy(),
        "p_log10_diag_mae": torch.mean(torch.abs(_safe_log10_diag(results["P_pred"]) - _safe_log10_diag(results["P0"])), dim=-1).numpy(),
        "q_log10_diag_mae": torch.mean(torch.abs(_safe_log10_diag(results["Q_pred"]) - _safe_log10_diag(results["Q0"])), dim=-1).numpy(),
        "r_log10_diag_mae": torch.mean(torch.abs(_safe_log10_diag(results["R_pred"]) - _safe_log10_diag(results["R0"])), dim=-1).numpy(),
        "u_rmse": u_rmse,
    }
    for idx in range(results["P0"].shape[0]):
        row = {"val_index": idx}
        for key, values in rels.items():
            row[key] = float(values[idx])
        per_sample_rows.append(row)

    metrics["per_sample_rows"] = per_sample_rows
    return metrics


def _save_json(path: Path, data: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _write_csv(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(metrics: Dict) -> list[Dict]:
    rows = []
    for name in ["P", "Q", "R"]:
        item = metrics[name]
        rows.append({
            "matrix": name,
            "r2_global": item["r2_global"],
            "frob_rel_mean": item["frob_rel"].get("mean"),
            "frob_rel_median": item["frob_rel"].get("median"),
            "frob_rel_p95": item["frob_rel"].get("p95"),
            "mae_mean": item["mae"].get("mean"),
            "rmse_mean": item["rmse"].get("mean"),
            "r2_per_sample_mean": item["r2_per_sample"].get("mean"),
            "r2_per_sample_median": item["r2_per_sample"].get("median"),
            "log10_diag_mae_mean": item["log10_diag_mae"].get("mean"),
            "log10_diag_mae_median": item["log10_diag_mae"].get("median"),
            "log10_diag_rmse_mean": item["log10_diag_rmse"].get("mean"),
            "log10_diag_r2_global": item["log10_diag_r2_global"],
        })
    return rows


def _save_table_image(rows: list[Dict], out_path: Path, title: str) -> None:
    labels = list(rows[0].keys())
    cell_text = []
    for row in rows:
        formatted = []
        for key in labels:
            value = row[key]
            if isinstance(value, (float, np.floating)):
                formatted.append(f"{value:.4g}")
            else:
                formatted.append(str(value))
        cell_text.append(formatted)

    fig_w = max(8, 1.25 * len(labels))
    fig_h = max(2.2, 0.45 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=13, pad=12)
    table = ax.table(cellText=cell_text, colLabels=labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_training_curves(history_path: Path, out_dir: Path) -> None:
    if not history_path.exists():
        return
    with history_path.open("r", encoding="utf-8") as f:
        history = json.load(f)
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for key, label in [("val_rel_p", "P"), ("val_rel_q", "Q"), ("val_rel_r", "R")]:
        axes[1].plot(epochs, [h[key] for h in history], label=label)
    axes[1].set_title("Validation Frobenius Relative Error")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=180)
    plt.close(fig)


def plot_error_histograms(metrics: Dict, out_dir: Path) -> None:
    rows = metrics["per_sample_rows"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, title in zip(axes, ["p_rel", "q_rel", "r_rel"], ["P", "Q", "R"]):
        values = np.array([r[key] for r in rows], dtype=np.float64)
        ax.hist(values, bins=35, color="#4c78a8", edgecolor="white", alpha=0.9)
        ax.axvline(np.median(values), color="black", linestyle="--", label=f"median={np.median(values):.3f}")
        ax.set_title(f"{title} relative error")
        ax.set_xlabel("Frobenius relative error")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "error_histograms_pqr.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, title in zip(
        axes,
        ["p_log10_diag_mae", "q_log10_diag_mae", "r_log10_diag_mae"],
        ["P", "Q", "R"],
    ):
        values = np.array([r[key] for r in rows], dtype=np.float64)
        ax.hist(values, bins=35, color="#59a14f", edgecolor="white", alpha=0.9)
        ax.axvline(np.median(values), color="black", linestyle="--", label=f"median={np.median(values):.3f}")
        ax.set_title(f"{title} log10 diagonal MAE")
        ax.set_xlabel("mean absolute log10 error")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "error_histograms_log10_diag_pqr.png", dpi=180)
    plt.close(fig)


def plot_parity(results: Dict[str, torch.Tensor], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, name in zip(axes, ["P", "Q", "R"]):
        target = results[f"{name}0"].numpy().reshape(-1)
        pred = results[f"{name}_pred"].numpy().reshape(-1)
        ax.scatter(target, pred, s=12, alpha=0.45, color="#f58518")
        lo = min(np.min(target), np.min(pred))
        hi = max(np.max(target), np.max(pred))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax.set_title(f"{name}: predicted vs target")
        ax.set_xlabel("Target")
        ax.set_ylabel("Prediction")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "parity_pqr_entries.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, name in zip(axes, ["P", "Q", "R"]):
        target = _safe_log10_diag(results[f"{name}0"]).numpy().reshape(-1)
        pred = _safe_log10_diag(results[f"{name}_pred"]).numpy().reshape(-1)
        ax.scatter(target, pred, s=12, alpha=0.45, color="#59a14f")
        lo = min(np.min(target), np.min(pred))
        hi = max(np.max(target), np.max(pred))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax.set_title(f"{name}: predicted vs target log10 diagonal")
        ax.set_xlabel("Target log10 diag")
        ax.set_ylabel("Prediction log10 diag")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "parity_log10_diag_pqr.png", dpi=180)
    plt.close(fig)


def plot_matrix_examples(results: Dict[str, torch.Tensor], metrics: Dict, out_dir: Path, n_examples: int) -> None:
    rows = metrics["per_sample_rows"]
    score = np.array([r["p_rel"] + r["q_rel"] + r["r_rel"] for r in rows], dtype=np.float64)
    if score.size == 0:
        return
    order = np.argsort(score)
    chosen = list(order[: max(1, n_examples // 2)])
    chosen += list(order[-max(1, n_examples - len(chosen)):])

    for matrix_name in ["P", "Q", "R"]:
        fig, axes = plt.subplots(len(chosen), 3, figsize=(9, 2.6 * len(chosen)))
        if len(chosen) == 1:
            axes = axes[None, :]
        for row_i, sample_i in enumerate(chosen):
            gt = results[f"{matrix_name}0"][sample_i].numpy()
            pred = results[f"{matrix_name}_pred"][sample_i].numpy()
            err = np.abs(pred - gt)
            vmax = max(float(np.max(np.abs(gt))), float(np.max(np.abs(pred))), 1e-8)
            panels = [(gt, "target"), (pred, "predicted"), (err, "abs error")]
            for col_i, (mat, title) in enumerate(panels):
                cmap = "Reds" if title == "abs error" else "RdBu_r"
                vmin = 0 if title == "abs error" else -vmax
                vmax_i = float(np.max(err)) if title == "abs error" else vmax
                im = axes[row_i, col_i].imshow(mat, cmap=cmap, vmin=vmin, vmax=max(vmax_i, 1e-8))
                axes[row_i, col_i].set_title(f"sample {sample_i} {title}", fontsize=9)
                axes[row_i, col_i].set_xticks([])
                axes[row_i, col_i].set_yticks([])
                for (rr, cc), value in np.ndenumerate(mat):
                    axes[row_i, col_i].text(cc, rr, f"{value:.2g}", ha="center", va="center", fontsize=7)
                fig.colorbar(im, ax=axes[row_i, col_i], fraction=0.046)
        fig.suptitle(f"{matrix_name} matrix examples: best and worst validation samples", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / f"{matrix_name.lower()}_matrix_examples.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Transformer-only PQR model on motor dataset.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_examples", type=int, default=6)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    run_dir = checkpoint.parent
    config = _load_config(checkpoint)
    data_root = args.data_root or config["data_root"]
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "eval_motor_pqr"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    manifest = get_manifest(data_root, target_scale=config.get("target_scale", "none"))

    ds = MPCDataset(
        data_root,
        drop_duplicate_seq_feature=True,
        target_scale=config.get("target_scale", "none"),
    )
    ds.set_normalization_stats(*_load_normalization(run_dir))
    target_mean, target_std = _load_target_normalization(run_dir)
    _, val_idx = split_indices(len(ds), float(config.get("val_frac", 0.1)), int(config.get("seed", 42)))
    loader = DataLoader(Subset(ds, val_idx.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = HybridControllerModel(
        d_in=manifest["d_seq"],
        d_static=manifest["d_static"],
        d_model=int(config["d_model"]),
        n=manifest["n"],
        m=manifest["m"],
        mamba_layers=int(config.get("mamba_layers", 0)),
        tx_layers=int(config["tx_layers"]),
        tx_heads=int(config["tx_heads"]),
        dropout=float(config["dropout"]),
        eps=float(config["eps"]),
        predict_u=True,
        predict_log_diag=(config.get("target_mode") == "log_diag"),
        predict_qr_log_diag=(config.get("target_mode") == "qr_log_diag"),
        predict_motor_struct=(config.get("target_mode") == "motor_struct"),
        arch=config.get("arch", "transformer"),
        seq_pool=config.get("seq_pool", "mean"),
        patch_len=int(config.get("patch_len", 12)),
        patch_stride=int(config.get("patch_stride", 6)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    results = collect_predictions(
        model,
        loader,
        device,
        target_mode=config.get("target_mode", "full_psd"),
        target_mean=target_mean,
        target_std=target_std,
    )
    metrics = compute_metrics(results)
    summary_rows = _summary_rows(metrics)

    result_json = {
        "checkpoint": str(checkpoint),
        "data_root": data_root,
        "out_dir": str(out_dir),
        "epoch": ckpt.get("epoch"),
        "num_val_samples": int(results["P0"].shape[0]),
        "manifest": manifest,
        "metrics": {k: v for k, v in metrics.items() if k != "per_sample_rows"},
    }
    _save_json(out_dir / "eval_results.json", result_json)
    _write_csv(out_dir / "summary_metrics.csv", summary_rows)
    _write_csv(out_dir / "per_sample_metrics.csv", metrics["per_sample_rows"])
    _save_table_image(summary_rows, out_dir / "summary_metrics_table.png", "PQR Evaluation Summary")

    plot_training_curves(run_dir / "history.json", out_dir)
    plot_error_histograms(metrics, out_dir)
    plot_parity(results, out_dir)
    plot_matrix_examples(results, metrics, out_dir, args.num_examples)

    print(f"[done] val_samples={results['P0'].shape[0]} output={out_dir}")
    for row in summary_rows:
        print(
            f"{row['matrix']}: R2_global={row['r2_global']:.4f} "
            f"rel_mean={row['frob_rel_mean']:.4f} rel_median={row['frob_rel_median']:.4f} "
            f"rel_p95={row['frob_rel_p95']:.4f}"
        )


if __name__ == "__main__":
    main()
