from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class MPCDataset(Dataset):
    """
    Dataset for the synthetic MPC PQR tuning problem.

    Each returned item is a dict with:
        - tokens_seq:    (T, d_seq)
        - tokens_static: (d_static,)
        - P0:            (n, n)
        - Q0:            (n, n)
        - R0:            (m, m)
        - u:             (T, m)
    """

    def __init__(
        self,
        root_dir: str,
        use_npz_if_available: bool = True,
        require_closed_loop: bool = True,
        dtype: torch.dtype = torch.float32,
        drop_duplicate_seq_feature: bool = False,
        duplicate_seq_feature_idx: int = 3,
    ) -> None:
        super().__init__()

        self.root_dir = Path(root_dir)
        self.samples_dir = self.root_dir / "samples"
        self.use_npz_if_available = use_npz_if_available
        self.require_closed_loop = require_closed_loop
        self.dtype = dtype

        self.drop_duplicate_seq_feature = drop_duplicate_seq_feature
        self.duplicate_seq_feature_idx = duplicate_seq_feature_idx

        # normalization stats; set later from train.py
        self.seq_mean: Optional[np.ndarray] = None
        self.seq_std: Optional[np.ndarray] = None
        self.static_mean: Optional[np.ndarray] = None
        self.static_std: Optional[np.ndarray] = None

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root_dir does not exist: {self.root_dir}")

        if not self.samples_dir.exists():
            raise FileNotFoundError(f"Samples directory not found: {self.samples_dir}")

        self.sample_dirs = self._discover_samples()

        if len(self.sample_dirs) == 0:
            raise RuntimeError(f"No valid samples found in {self.samples_dir}")

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

    def _discover_samples(self) -> List[Path]:
        sample_dirs: List[Path] = []

        for sample_dir in sorted(self.samples_dir.glob("sample_*")):
            meta_path = sample_dir / "meta.json"
            if not meta_path.exists():
                continue

            if self.require_closed_loop and not (sample_dir / "closed_loop.npz").exists():
                continue

            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                continue

            if not meta.get("success", False):
                continue

            if "signals" not in meta or "target" not in meta:
                continue

            signals = meta["signals"]
            target = meta["target"]

            if "tokens_seq" not in signals or "tokens_static" not in signals:
                continue

            if "P_raw" not in target or "Q_raw" not in target or "R_raw" not in target:
                continue

            sample_dirs.append(sample_dir)

        return sample_dirs

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def _load_meta(self, sample_dir: Path) -> Dict:
        meta_path = sample_dir / "meta.json"
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_signals(self, sample_dir: Path, meta: Dict) -> Dict[str, np.ndarray]:
        signals_npz = sample_dir / "signals.npz"

        if self.use_npz_if_available and signals_npz.exists():
            data = np.load(signals_npz)
            out = {
                "tokens_seq": np.array(data["tokens_seq"], dtype=np.float32),
                "tokens_static": np.array(data["tokens_static"], dtype=np.float32),
            }
            return out

        signals = meta["signals"]
        return {
            "tokens_seq": np.array(signals["tokens_seq"], dtype=np.float32),
            "tokens_static": np.array(signals["tokens_static"], dtype=np.float32),
        }

    def _load_closed_loop(self, sample_dir: Path, meta: Dict) -> Dict[str, np.ndarray]:
        closed_loop_npz = sample_dir / "closed_loop.npz"

        if self.use_npz_if_available and closed_loop_npz.exists():
            data = np.load(closed_loop_npz)
            u_traj = np.array(data["u_traj"], dtype=np.float32)
            return {
                "u_traj": u_traj,
            }

        if "closed_loop" not in meta:
            raise KeyError(f"closed_loop data missing for sample: {sample_dir.name}")

        closed_loop = meta["closed_loop"]
        return {
            "u_traj": np.array(closed_loop["u_traj"], dtype=np.float32),
        }

    def _maybe_drop_duplicate_seq_feature(self, tokens_seq: np.ndarray) -> np.ndarray:
        if not self.drop_duplicate_seq_feature:
            return tokens_seq

        if tokens_seq.ndim != 2:
            raise ValueError(f"tokens_seq must have shape (T, d), got {tokens_seq.shape}")

        d = tokens_seq.shape[1]
        idx = self.duplicate_seq_feature_idx

        if idx < 0 or idx >= d:
            raise IndexError(f"duplicate_seq_feature_idx={idx} out of range for d={d}")

        keep = [j for j in range(d) if j != idx]
        return tokens_seq[:, keep]

    def _maybe_normalize_seq(self, tokens_seq: np.ndarray) -> np.ndarray:
        if self.seq_mean is None or self.seq_std is None:
            return tokens_seq
        return (tokens_seq - self.seq_mean[None, :]) / self.seq_std[None, :]

    def _maybe_normalize_static(self, tokens_static: np.ndarray) -> np.ndarray:
        if self.static_mean is None or self.static_std is None:
            return tokens_static
        return (tokens_static - self.static_mean) / self.static_std

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_dir = self.sample_dirs[idx]
        meta = self._load_meta(sample_dir)

        signals = self._load_signals(sample_dir, meta)
        closed_loop = self._load_closed_loop(sample_dir, meta)

        target = meta["target"]

        tokens_seq = signals["tokens_seq"]              # (T, d_seq)
        tokens_static = signals["tokens_static"]        # (d_static,)
        P0 = np.array(target["P_raw"], dtype=np.float32)
        Q0 = np.array(target["Q_raw"], dtype=np.float32)
        R0 = np.array(target["R_raw"], dtype=np.float32)

        u = closed_loop["u_traj"]                       # (T,) or (T,1)

        # ensure control has shape (T, m)
        if u.ndim == 1:
            u = u[:, None]

        # optional feature cleanup
        tokens_seq = self._maybe_drop_duplicate_seq_feature(tokens_seq)

        # optional normalization
        tokens_seq = self._maybe_normalize_seq(tokens_seq)
        tokens_static = self._maybe_normalize_static(tokens_static)

        item = {
            "tokens_seq": torch.tensor(tokens_seq, dtype=self.dtype),
            "tokens_static": torch.tensor(tokens_static, dtype=self.dtype),
            "P0": torch.tensor(P0, dtype=self.dtype),
            "Q0": torch.tensor(Q0, dtype=self.dtype),
            "R0": torch.tensor(R0, dtype=self.dtype),
            "u": torch.tensor(u, dtype=self.dtype),
        }

        return item


if __name__ == "__main__":
    ds = MPCDataset(
        root_dir="mpc_pqr_dataset_streaming_debug",
        drop_duplicate_seq_feature=True,
        duplicate_seq_feature_idx=3,
    )
    print("Dataset size:", len(ds))

    sample = ds[0]
    for k, v in sample.items():
        print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}")