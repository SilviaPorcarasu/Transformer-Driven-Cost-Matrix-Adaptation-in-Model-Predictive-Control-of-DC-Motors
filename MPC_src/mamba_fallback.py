"""
Fast CPU/MPS fallback for the Mamba block.
Uses GRU + depthwise Conv1D + gating to approximate the Mamba architecture.
Same constructor signature and forward interface as mamba_ssm.Mamba.

When running on CUDA with mamba-ssm installed, the real Mamba is used instead.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mamba(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)

        # Input projection -> x and gate z
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Causal depthwise convolution
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )
        self.act = nn.SiLU()

        # GRU for sequential modeling (replaces selective scan)
        self.gru = nn.GRU(
            input_size=self.d_inner,
            hidden_size=self.d_inner,
            num_layers=1,
            batch_first=True,
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, d_model) -> (B, T, d_model)"""
        B, T, _ = x.shape

        # 1. Input projection
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)       # each (B, T, d_inner)

        # 2. Causal depthwise conv + activation
        x_conv = x_inner.transpose(1, 2)        # (B, d_inner, T)
        x_conv = self.conv1d(x_conv)[:, :, :T]  # causal trim
        x_inner = self.act(x_conv).transpose(1, 2)

        # 3. GRU (replaces selective scan)
        y, _ = self.gru(x_inner)                 # (B, T, d_inner)

        # 4. Gate with z
        y = y * self.act(z)

        # 5. Output projection
        return self.out_proj(y)
