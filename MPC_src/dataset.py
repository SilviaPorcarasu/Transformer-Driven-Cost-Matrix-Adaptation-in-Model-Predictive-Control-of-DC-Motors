from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class MPCDataset(Dataset):
    """Dataset reader for the motor-CC PQR sample-folder format."""

    def __init__(
        self,
        root_dir: str,
        use_npz_if_available: bool = True,
        require_closed_loop: bool = True,
        dtype: torch.dtype = torch.float32,
        drop_duplicate_seq_feature: bool = True,
        duplicate_seq_feature_idx: int = 3,
        target_scale: str = "none",
    ) -> None:
        super().__init__()

        self.root_dir = Path(root_dir)
        self.samples_dir = self.root_dir / "samples"
        self.use_npz_if_available = use_npz_if_available
        self.require_closed_loop = require_closed_loop
        self.dtype = dtype
        self.drop_duplicate_seq_feature = drop_duplicate_seq_feature
        self.duplicate_seq_feature_idx = duplicate_seq_feature_idx
        if target_scale not in {"none", "trace", "r"}:
            raise ValueError("target_scale must be one of: none, trace, r")
        self.target_scale = target_scale

        self.seq_mean: Optional[np.ndarray] = None
        self.seq_std: Optional[np.ndarray] = None
        self.static_mean: Optional[np.ndarray] = None
        self.static_std: Optional[np.ndarray] = None

        if not self.samples_dir.exists():
            raise FileNotFoundError(f"Samples directory not found: {self.samples_dir}")

        self.sample_dirs = self._discover_samples()
        if not self.sample_dirs:
            raise RuntimeError(f"No valid PQR motor samples found in {self.samples_dir}")

    def _discover_samples(self) -> List[Path]:
        sample_dirs: List[Path] = []
        for sample_dir in sorted(self.samples_dir.glob("sample_*")):
            meta_path = sample_dir / "meta.json"
            signals_path = sample_dir / "signals.npz"
            if not meta_path.exists() or not signals_path.exists():
                continue
            if self.require_closed_loop and not (sample_dir / "closed_loop.npz").exists():
                continue

            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                continue

            target = meta.get("target", {})
            signals = meta.get("signals", {})
            has_targets = all(k in target for k in ("P_raw", "Q_raw", "R_raw"))
            has_tokens = all(k in signals for k in ("tokens_seq", "tokens_static"))
            if meta.get("success", False) and has_targets and has_tokens:
                sample_dirs.append(sample_dir)
        return sample_dirs

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def set_normalization_stats(
        self,
        seq_mean: Optional[np.ndarray],
        seq_std: Optional[np.ndarray],
        static_mean: Optional[np.ndarray],
        static_std: Optional[np.ndarray],
    ) -> None:
        self.seq_mean = None if seq_mean is None else np.asarray(seq_mean, dtype=np.float32)
        self.seq_std = None if seq_std is None else np.asarray(seq_std, dtype=np.float32)
        self.static_mean = None if static_mean is None else np.asarray(static_mean, dtype=np.float32)
        self.static_std = None if static_std is None else np.asarray(static_std, dtype=np.float32)

    def _load_meta(self, sample_dir: Path) -> Dict:
        with (sample_dir / "meta.json").open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_signals(self, sample_dir: Path, meta: Dict) -> Dict[str, np.ndarray]:
        signals_npz = sample_dir / "signals.npz"
        if self.use_npz_if_available and signals_npz.exists():
            data = np.load(signals_npz)
            return {
                "tokens_seq": np.asarray(data["tokens_seq"], dtype=np.float32),
                "tokens_static": np.asarray(data["tokens_static"], dtype=np.float32),
            }

        signals = meta["signals"]
        return {
            "tokens_seq": np.asarray(signals["tokens_seq"], dtype=np.float32),
            "tokens_static": np.asarray(signals["tokens_static"], dtype=np.float32),
        }

    def _load_closed_loop(self, sample_dir: Path, meta: Dict) -> np.ndarray:
        closed_loop_npz = sample_dir / "closed_loop.npz"
        if self.use_npz_if_available and closed_loop_npz.exists():
            data = np.load(closed_loop_npz)
            return np.asarray(data["u_traj"], dtype=np.float32)
        return np.asarray(meta["closed_loop"]["u_traj"], dtype=np.float32)

    def _maybe_drop_duplicate_seq_feature(self, tokens_seq: np.ndarray) -> np.ndarray:
        if not self.drop_duplicate_seq_feature:
            return tokens_seq
        keep = [j for j in range(tokens_seq.shape[1]) if j != self.duplicate_seq_feature_idx]
        return tokens_seq[:, keep]

    def _maybe_normalize_seq(self, tokens_seq: np.ndarray) -> np.ndarray:
        if self.seq_mean is None or self.seq_std is None:
            return tokens_seq
        return (tokens_seq - self.seq_mean[None, :]) / self.seq_std[None, :]

    def _maybe_normalize_static(self, tokens_static: np.ndarray) -> np.ndarray:
        if self.static_mean is None or self.static_std is None:
            return tokens_static
        return (tokens_static - self.static_mean) / self.static_std

    def _scale_targets(
        self,
        P0: np.ndarray,
        Q0: np.ndarray,
        R0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if self.target_scale == "none":
            scale = 1.0
        elif self.target_scale == "trace":
            scale = float(np.trace(Q0) + np.trace(R0))
        else:
            scale = float(np.trace(R0))

        scale = max(scale, 1e-12)
        return P0 / scale, Q0 / scale, R0 / scale, scale

    def raw_tokens_for_stats(self, idx: int) -> Dict[str, np.ndarray]:
        sample_dir = self.sample_dirs[idx]
        meta = self._load_meta(sample_dir)
        signals = self._load_signals(sample_dir, meta)
        return {
            "tokens_seq": self._maybe_drop_duplicate_seq_feature(signals["tokens_seq"]),
            "tokens_static": signals["tokens_static"],
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_dir = self.sample_dirs[idx]
        meta = self._load_meta(sample_dir)
        signals = self._load_signals(sample_dir, meta)
        target = meta["target"]
        model = meta["model"]

        tokens_seq = self._maybe_drop_duplicate_seq_feature(signals["tokens_seq"])
        tokens_static = signals["tokens_static"]
        tokens_seq = self._maybe_normalize_seq(tokens_seq)
        tokens_static = self._maybe_normalize_static(tokens_static)

        u = self._load_closed_loop(sample_dir, meta)
        if u.ndim == 1:
            u = u[:, None]

        P0 = np.asarray(target["P_raw"], dtype=np.float32)
        Q0 = np.asarray(target["Q_raw"], dtype=np.float32)
        R0 = np.asarray(target["R_raw"], dtype=np.float32)
        P0, Q0, R0, target_scale = self._scale_targets(P0, Q0, R0)

        return {
            "tokens_seq": torch.tensor(tokens_seq, dtype=self.dtype),
            "tokens_static": torch.tensor(tokens_static, dtype=self.dtype),
            "P0": torch.tensor(P0, dtype=self.dtype),
            "Q0": torch.tensor(Q0, dtype=self.dtype),
            "R0": torch.tensor(R0, dtype=self.dtype),
            "target_scale": torch.tensor(target_scale, dtype=self.dtype),
            "A": torch.tensor(np.asarray(model["A_d"], dtype=np.float32), dtype=self.dtype),
            "B": torch.tensor(np.asarray(model["B_d"], dtype=np.float32), dtype=self.dtype),
            "u": torch.tensor(u, dtype=self.dtype),
        }


def get_manifest(root_dir: str, target_scale: str = "none") -> Dict[str, int]:
    ds = MPCDataset(root_dir, target_scale=target_scale)
    first = ds[0]
    return {
        "n": int(first["P0"].shape[0]),
        "m": int(first["R0"].shape[0]),
        "T": int(first["tokens_seq"].shape[0]),
        "d_seq": int(first["tokens_seq"].shape[1]),
        "d_static": int(first["tokens_static"].shape[0]),
        "num_samples": len(ds),
    }
