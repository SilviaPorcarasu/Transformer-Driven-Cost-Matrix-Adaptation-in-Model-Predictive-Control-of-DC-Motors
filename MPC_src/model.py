import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError:
    from mamba_fallback import Mamba


def lower_unit_from_vec(v: torch.Tensor, n: int) -> torch.Tensor:
    """Build unit lower-triangular matrices from off-diagonal vectors."""
    bsz = v.shape[0]
    l = torch.eye(n, device=v.device, dtype=v.dtype).unsqueeze(0).repeat(bsz, 1, 1)
    idx = torch.tril_indices(row=n, col=n, offset=-1, device=v.device)
    if idx.numel() > 0:
        l[:, idx[0], idx[1]] = v
    return l


class PSDHead(nn.Module):
    """Predict a single PSD matrix with LDL^T parameterization."""

    def __init__(self, d_model: int, size: int, eps: float = 1e-3):
        super().__init__()
        self.size = size
        self.eps = eps
        off_dim = size * (size - 1) // 2
        diag_dim = size
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, off_dim + diag_dim),
        )
        self.off_dim = off_dim
        self.diag_dim = diag_dim

    def forward(self, h_global: torch.Tensor):
        out = self.mlp(h_global)
        off = out[:, :self.off_dim]
        diag = out[:, self.off_dim:]
        L = lower_unit_from_vec(off, self.size)
        d = F.softplus(diag) + self.eps
        return L @ torch.diag_embed(d) @ L.transpose(-1, -2)


class QRHeadPSD(nn.Module):
    """Predict PSD Q and R with an LDL^T parameterization.

    Using L (unit lower-triangular) and positive diagonal D avoids the
    near-zero-gradient issue of pure L @ L^T when outputs start close to 0.
    """

    def __init__(self, d_model: int, n: int, m: int, eps: float = 1e-3):
        super().__init__()
        self.n = n
        self.m = m
        self.eps = eps

        self.q_off_dim = n * (n - 1) // 2
        self.q_diag_dim = n
        self.r_off_dim = m * (m - 1) // 2
        self.r_diag_dim = m

        out_dim = self.q_off_dim + self.q_diag_dim + self.r_off_dim + self.r_diag_dim

        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, h_global: torch.Tensor):
        out = self.mlp(h_global)

        i = 0
        q_off = out[:, i: i + self.q_off_dim]
        i += self.q_off_dim
        q_diag = out[:, i: i + self.q_diag_dim]
        i += self.q_diag_dim
        r_off = out[:, i: i + self.r_off_dim]
        i += self.r_off_dim
        r_diag = out[:, i: i + self.r_diag_dim]

        lq = lower_unit_from_vec(q_off, self.n)
        lr = lower_unit_from_vec(r_off, self.m)

        # Strictly positive diagonals -> PSD/PD matrices with healthy gradients at init.
        dq = F.softplus(q_diag) + self.eps
        dr = F.softplus(r_diag) + self.eps

        Q = lq @ torch.diag_embed(dq) @ lq.transpose(-1, -2)
        R = lr @ torch.diag_embed(dr) @ lr.transpose(-1, -2)
        return Q, R



class TransformerStack(nn.Module):
    # stack of transformer encoder layer
    def __init__(self, d_model, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, x):
        return self.enc(x)


class MambaStack(nn.Module):
    def __init__(self, d_model, n_layers=4, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        for layer in self.layers:
            x = x + layer(x)  # residual
        return self.norm(x)


class HybridControllerModel(nn.Module):
    def __init__(
        self,
        d_in=8,  # tokens [y(n), u_prev(m), dt(1), sat(1)]
        d_model=256,
        n=4,
        m=2,
        n_max=None,  # compatibility with older callers
        m_max=None,  # compatibility with older callers
        mamba_layers=4,
        tx_layers=2,
        tx_heads=4,
        dropout=0.1,
        eps=1e-3,
    ):
        super().__init__()

        if n_max is not None:
            n = n_max
        if m_max is not None:
            m = m_max

        self.n = int(n)
        self.m = int(m)

        self.proj = nn.Linear(d_in, d_model)
        self.mamba = MambaStack(d_model, n_layers=mamba_layers)
        self.bridge = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.transformer = TransformerStack(
            d_model=d_model,
            n_layers=tx_layers,
            n_heads=tx_heads,
            dropout=dropout,
        )
        self.qr_head = QRHeadPSD(d_model, n=self.n, m=self.m, eps=eps)
        self.p_head = PSDHead(d_model, size=self.n, eps=eps)

    def forward(self, tokens):
        h = self.proj(tokens)       # (B,T,d_in) -> (B,T,d_model)
        h = self.mamba(h)           # (B,T,d_model)
        h = self.bridge(h)          # (B,T,d_model)
        h = self.transformer(h)     # (B,T,d_model)
        h_global = h.mean(dim=1)    # (B,T,d_model) -> (B,d_model)
        q_pred, r_pred = self.qr_head(h_global)  # Q: (B,n,n), R: (B,m,m)
        p_pred = self.p_head(h_global)            # P: (B,n,n)
        return p_pred, q_pred, r_pred


# test
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dataset import LQRDataset

    ds = LQRDataset(root_dir="MPC_dataset/synthetic_lqr_data")
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)

    batch = next(iter(dl))
    tokens = batch["tokens"]
    d_in = tokens.shape[-1]

    model = HybridControllerModel(d_in=d_in, d_model=256, n=ds.n, m=ds.m)
    pp, qp, rp = model(tokens)

    print(f"P_pred: {pp.shape}")   # (32, n, n)
    print(f"Q_pred: {qp.shape}")   # (32, n, n)
    print(f"R_pred: {rp.shape}")   # (32, m, m)
