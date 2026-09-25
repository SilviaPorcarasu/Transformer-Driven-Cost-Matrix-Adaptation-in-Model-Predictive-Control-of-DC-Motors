import torch
import torch.nn as nn
import torch.nn.functional as F


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


class LogDiagPQRHead(nn.Module):
    """Predict log10 diagonals for P, Q and R."""

    def __init__(self, d_model: int, n: int, m: int):
        super().__init__()
        self.n = n
        self.m = m
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n + n + m),
        )

    def forward(self, h_global: torch.Tensor):
        out = self.net(h_global)
        p_log = out[:, :self.n]
        q_log = out[:, self.n:self.n + self.n]
        r_log = out[:, self.n + self.n:]
        return p_log, q_log, r_log


class LogDiagQRHead(nn.Module):
    """Predict log10 diagonals for Q and R only."""

    def __init__(self, d_model: int, n: int, m: int):
        super().__init__()
        self.n = n
        self.m = m
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n + m),
        )

    def forward(self, h_global: torch.Tensor):
        out = self.net(h_global)
        q_log = out[:, :self.n]
        r_log = out[:, self.n:]
        return q_log, r_log


class MotorStructuredPQRHead(nn.Module):
    """Structured PSD head for the 2-state, 1-input DC motor case.

    The head predicts log10 diagonal entries plus bounded correlations for
    the off-diagonal terms. This keeps P and Q positive definite while using
    only the degrees of freedom that matter for n=2, m=1.
    """

    def __init__(self, d_model: int, eps: float = 1e-6, log_min: float = -8.0, log_max: float = 4.0):
        super().__init__()
        self.eps = eps
        self.log_min = log_min
        self.log_max = log_max
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 8),
        )

    def _diag_from_log(self, log_diag: torch.Tensor) -> torch.Tensor:
        return torch.pow(10.0, log_diag.clamp(self.log_min, self.log_max)) + self.eps

    def _matrix_2x2(self, log_d1: torch.Tensor, log_d2: torch.Tensor, rho_raw: torch.Tensor) -> torch.Tensor:
        d1 = self._diag_from_log(log_d1)
        d2 = self._diag_from_log(log_d2)
        rho = 0.999 * torch.tanh(rho_raw)
        off = rho * torch.sqrt(d1 * d2)

        out = torch.zeros(log_d1.shape[0], 2, 2, device=log_d1.device, dtype=log_d1.dtype)
        out[:, 0, 0] = d1
        out[:, 1, 1] = d2
        out[:, 0, 1] = off
        out[:, 1, 0] = off
        return out

    def forward(self, h_global: torch.Tensor):
        out = self.net(h_global)
        p = self._matrix_2x2(out[:, 0], out[:, 1], out[:, 2])
        q = self._matrix_2x2(out[:, 3], out[:, 4], out[:, 5])
        r = self._diag_from_log(out[:, 6]).view(-1, 1, 1)
        return p, q, r


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


class AttentionPool(nn.Module):
    """Learned weighted pooling over sequence tokens."""

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(h), dim=1)
        return torch.sum(weights * h, dim=1)


class PatchTimeEncoder(nn.Module):
    """Patch-based encoder for short multivariate motor time series."""

    def __init__(
        self,
        d_in: int,
        d_model: int,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        patch_len: int = 12,
        patch_stride: int = 6,
        max_patches: int = 128,
    ):
        super().__init__()
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.max_patches = int(max_patches)
        if self.patch_len <= 0 or self.patch_stride <= 0:
            raise ValueError("patch_len and patch_stride must be positive")

        self.patch_proj = nn.Linear(self.patch_len * d_in, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_patches, d_model))
        self.input_norm = nn.LayerNorm(d_model)
        self.transformer = TransformerStack(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] < self.patch_len:
            pad = self.patch_len - tokens.shape[1]
            tokens = F.pad(tokens, (0, 0, 0, pad))

        patches = tokens.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        patches = patches.transpose(-1, -2).contiguous()
        patches = patches.reshape(patches.shape[0], patches.shape[1], -1)
        if patches.shape[1] > self.max_patches:
            raise ValueError(f"Patch count {patches.shape[1]} exceeds max_patches={self.max_patches}")

        h = self.patch_proj(patches)
        h = h + self.pos_embedding[:, :h.shape[1], :]
        h = self.input_norm(h)
        h = self.transformer(h)
        return h.mean(dim=1)


class HybridControllerModel(nn.Module):
    def __init__(
        self,
        d_in=8,  # tokens [y(n), u_prev(m), dt(1), sat(1)]
        d_static=0,
        d_model=256,
        n=4,
        m=2,
        n_max=None,  # compatibility with older callers
        m_max=None,  # compatibility with older callers
        mamba_layers=0,  # deprecated compatibility argument; Transformer-only model
        tx_layers=2,
        tx_heads=4,
        dropout=0.1,
        eps=1e-3,
        predict_u=False,
        predict_log_diag=False,
        predict_qr_log_diag=False,
        predict_motor_struct=False,
        max_seq_len=512,
        arch="transformer",
        seq_pool="mean",
        patch_len=12,
        patch_stride=6,
    ):
        super().__init__()

        if n_max is not None:
            n = n_max
        if m_max is not None:
            m = m_max

        self.n = int(n)
        self.m = int(m)
        self.d_static = int(d_static or 0)
        self.predict_u = bool(predict_u)
        self.predict_log_diag = bool(predict_log_diag)
        self.predict_qr_log_diag = bool(predict_qr_log_diag)
        self.predict_motor_struct = bool(predict_motor_struct)
        self.max_seq_len = int(max_seq_len)
        self.arch = str(arch)
        self.seq_pool = str(seq_pool)

        if self.arch == "patchtst":
            max_patches = max(1, (self.max_seq_len - int(patch_len)) // int(patch_stride) + 1)
            self.seq_encoder = PatchTimeEncoder(
                d_in=d_in,
                d_model=d_model,
                n_layers=tx_layers,
                n_heads=tx_heads,
                dropout=dropout,
                patch_len=patch_len,
                patch_stride=patch_stride,
                max_patches=max_patches,
            )
        elif self.arch == "transformer":
            self.proj = nn.Linear(d_in, d_model)
            self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_seq_len, d_model))
            self.input_norm = nn.LayerNorm(d_model)
            self.transformer = TransformerStack(
                d_model=d_model,
                n_layers=tx_layers,
                n_heads=tx_heads,
                dropout=dropout,
            )
            if self.seq_pool == "attention":
                self.sequence_pool = AttentionPool(d_model)
            elif self.seq_pool == "mean":
                self.sequence_pool = None
            else:
                raise ValueError(f"Unknown sequence pooling: {self.seq_pool}")
        else:
            raise ValueError(f"Unknown architecture: {self.arch}")
        if self.predict_motor_struct:
            if self.n != 2 or self.m != 1:
                raise ValueError("predict_motor_struct requires n=2 and m=1")
            self.motor_struct_head = MotorStructuredPQRHead(d_model, eps=eps)
        elif self.predict_qr_log_diag:
            self.qr_log_head = LogDiagQRHead(d_model, n=self.n, m=self.m)
        elif self.predict_log_diag:
            self.pqr_log_head = LogDiagPQRHead(d_model, n=self.n, m=self.m)
        else:
            self.qr_head = QRHeadPSD(d_model, n=self.n, m=self.m, eps=eps)
            self.p_head = PSDHead(d_model, size=self.n, eps=eps)

        if self.d_static > 0:
            self.static_encoder = nn.Sequential(
                nn.LayerNorm(self.d_static),
                nn.Linear(self.d_static, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.global_fusion = nn.Sequential(
                nn.LayerNorm(2 * d_model),
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

        if self.predict_u:
            self.policy_fusion = nn.Sequential(
                nn.LayerNorm(2 * d_model if self.d_static > 0 else d_model),
                nn.Linear(2 * d_model if self.d_static > 0 else d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.u_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, self.m),
            )

    def forward(self, tokens, tokens_static=None):
        if self.arch == "patchtst":
            if tokens.shape[1] > self.max_seq_len:
                raise ValueError(f"Sequence length {tokens.shape[1]} exceeds max_seq_len={self.max_seq_len}")
            h = None
            h_global = self.seq_encoder(tokens)
        else:
            h = self.proj(tokens)       # (B,T,d_in) -> (B,T,d_model)
            if h.shape[1] > self.max_seq_len:
                raise ValueError(f"Sequence length {h.shape[1]} exceeds max_seq_len={self.max_seq_len}")
            h = h + self.pos_embedding[:, :h.shape[1], :]
            h = self.input_norm(h)
            h = self.transformer(h)     # (B,T,d_model)
            if self.sequence_pool is None:
                h_global = h.mean(dim=1)    # (B,T,d_model) -> (B,d_model)
            else:
                h_global = self.sequence_pool(h)

        h_static = None
        if self.d_static > 0:
            if tokens_static is None:
                raise ValueError("tokens_static is required when d_static > 0")
            h_static = self.static_encoder(tokens_static)
            h_global = self.global_fusion(torch.cat([h_global, h_static], dim=-1))

        if self.predict_motor_struct:
            p_pred, q_pred, r_pred = self.motor_struct_head(h_global)
        elif self.predict_qr_log_diag:
            q_pred, r_pred = self.qr_log_head(h_global)
            p_pred = None
        elif self.predict_log_diag:
            p_pred, q_pred, r_pred = self.pqr_log_head(h_global)
        else:
            q_pred, r_pred = self.qr_head(h_global)  # Q: (B,n,n), R: (B,m,m)
            p_pred = self.p_head(h_global)            # P: (B,n,n)

        if self.predict_u:
            if h is None:
                h = h_global.unsqueeze(1).expand(-1, tokens.shape[1], -1)
            if h_static is not None:
                h_static_seq = h_static.unsqueeze(1).expand(-1, h.shape[1], -1)
                h_policy = self.policy_fusion(torch.cat([h, h_static_seq], dim=-1))
            else:
                h_policy = self.policy_fusion(h)
            u_hat = self.u_head(h_policy)
            if self.predict_qr_log_diag:
                return q_pred, r_pred, u_hat
            return p_pred, q_pred, r_pred, u_hat

        if self.predict_qr_log_diag:
            return q_pred, r_pred
        return p_pred, q_pred, r_pred
