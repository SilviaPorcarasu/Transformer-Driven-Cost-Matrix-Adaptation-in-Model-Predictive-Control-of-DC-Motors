import glob
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def _dims_from_manifest(manifest):
    if "n" in manifest and "m" in manifest:
        return int(manifest["n"]), int(manifest["m"])
    if "n_max" in manifest and "m_max" in manifest:
        return int(manifest["n_max"]), int(manifest["m_max"])
    raise KeyError("manifest.json must contain either (n,m) or (n_max,m_max)")


class LQRDataset(Dataset):
    # sys_ids: optional set of system IDs to include (for train/val split by system)
    def __init__(self, root_dir="MPC_dataset/synthetic_lqr_data", sys_ids=None):
        pattern = root_dir.rstrip("/\\") + "/synthetic_shard_*.npz"
        self.shards = sorted(glob.glob(pattern))
        if len(self.shards) == 0:
            raise FileNotFoundError(f"Not found shards at pattern: {pattern}")

        manifest = get_manifest(root_dir)
        self.n, self.m = _dims_from_manifest(manifest)

        self.index = []
        for sp in self.shards:
            with np.load(sp, allow_pickle=True) as data:
                n_records = data["A"].shape[0]
                for i in range(n_records):
                    if sys_ids is not None:
                        meta = self._parse_meta(data["meta"][i])
                        if meta["sys_id"] not in sys_ids:
                            continue
                    self.index.append((sp, i))

        self._cache_path = None
        self._cache_data = None

    def __len__(self):
        return len(self.index)

    def _load_shard(self, shard_path):
        if self._cache_path == shard_path and self._cache_data is not None:
            return self._cache_data

        data = np.load(shard_path, allow_pickle=True)
        self._cache_path = shard_path
        self._cache_data = data
        return data

    @staticmethod
    def _parse_meta(meta_entry):
        if isinstance(meta_entry, (str, bytes)):
            s = meta_entry.decode("utf-8") if isinstance(meta_entry, bytes) else meta_entry
            return json.loads(s)
        return json.loads(str(meta_entry))

    def __getitem__(self, idx):
        shard_path, ep = self.index[idx]
        data = self._load_shard(shard_path)

        A = torch.from_numpy(data["A"][ep]).float()[: self.n, : self.n]
        B = torch.from_numpy(data["B"][ep]).float()[: self.n, : self.m]
        dt = torch.tensor(float(data["dt"][ep]), dtype=torch.float32)
        y = torch.from_numpy(data["y"][ep]).float()[:, : self.n]
        u = torch.from_numpy(data["u"][ep]).float()[:, : self.m]
        Q0 = torch.from_numpy(data["Q0"][ep]).float()[: self.n, : self.n]
        R0 = torch.from_numpy(data["R0"][ep]).float()[: self.m, : self.m]

        meta = self._parse_meta(data["meta"][ep])
        u_max_raw = meta.get("u_max", 1.0)
        if u_max_raw is None:
            u_max_raw = 1.0
        u_max = float(u_max_raw)

        u_prev = torch.zeros_like(u)
        u_prev[1:] = u[:-1]

        t_steps = y.shape[0]
        dt_feat = dt.repeat(t_steps).unsqueeze(-1)

        um = torch.tensor(u_max, dtype=torch.float32)
        sat_flag = (torch.abs(u) >= 0.999 * um).any(dim=-1, keepdim=True).float()

        # token = [y(n), u_prev(m), dt(1), sat_flag(1)]
        tokens = torch.cat([y, u_prev, dt_feat, sat_flag], dim=-1)

        return {
            "tokens": tokens,  # (T, n+m+2)
            "Q0": Q0,  # (n, n)
            "R0": R0,  # (m, m)
            "A": A,  # (n, n)
            "B": B,  # (n, m)
            "dt": dt,
            "u_max": um,
            "y": y,  # (T, n)
            "u": u,  # (T, m)
            "sat_flag": sat_flag,
            "sys_id": torch.tensor(int(meta["sys_id"]), dtype=torch.long),
            "episode_id": torch.tensor(int(meta["episode_id"]), dtype=torch.long),
        }


def get_manifest(root_dir="MPC_dataset/synthetic_lqr_data"):
    manifest_path = root_dir.rstrip("/\\") + "/manifest.json"
    with open(manifest_path) as f:
        return json.load(f)


def get_num_systems(root_dir="MPC_dataset/synthetic_lqr_data"):
    return int(get_manifest(root_dir)["num_systems"])


if __name__ == "__main__":
    ds = LQRDataset(root_dir="MPC_dataset/synthetic_lqr_data")
    print(f"Total episodes: {len(ds)}")
    print(f"n={ds.n}, m={ds.m}")

    ex = ds[0]
    print("\nSingle example:")
    print(f"  tokens: {ex['tokens'].shape}")  # (T, n+m+2)
    print(f"  Q0: {ex['Q0'].shape}")  # (n, n)
    print(f"  R0: {ex['R0'].shape}")  # (m, m)
    print(f"  sys_id: {ex['sys_id'].item()}")
    print(f"  episode_id: {ex['episode_id'].item()}")

    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
    batch = next(iter(dl))
    print("\nBatch example:")
    print(f"  tokens batch: {batch['tokens'].shape}")
    print(f"  Q0 batch: {batch['Q0'].shape}")
    print(f"  R0 batch: {batch['R0'].shape}")
