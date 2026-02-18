"""Brain4DCNN for 4th-order O-information tensors.

Architecture:
- 4D E2E: axis-wise 1D convolutions over four spatial axes.
- 4D E2N: region-centric aggregation from four index placements.
- N2G: permutation-invariant global average pooling to graph embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def _conv1d_along_axis_4d(x: torch.Tensor, conv: nn.Conv1d, axis: int) -> torch.Tensor:
    """Apply 1D convolution along one spatial axis of a 6D tensor."""
    B, M_in, C0, C1, C2, C3 = x.shape
    if axis == 0:
        x = x.permute(0, 3, 4, 5, 1, 2).contiguous()   # (B, C1, C2, C3, M_in, C0)
        batch, L = B * C1 * C2 * C3, C0
    elif axis == 1:
        x = x.permute(0, 2, 4, 5, 1, 3).contiguous()   # (B, C0, C2, C3, M_in, C1)
        batch, L = B * C0 * C2 * C3, C1
    elif axis == 2:
        x = x.permute(0, 2, 3, 5, 1, 4).contiguous()   # (B, C0, C1, C3, M_in, C2)
        batch, L = B * C0 * C1 * C3, C2
    else:
        x = x.permute(0, 2, 3, 4, 1, 5).contiguous()   # (B, C0, C1, C2, M_in, C3)
        batch, L = B * C0 * C1 * C2, C3
    x = x.reshape(batch, M_in, L)
    x = conv(x)   # (batch, M_out, L)
    M_out = x.shape[1]
    if axis == 0:
        x = x.reshape(B, C1, C2, C3, M_out, L).permute(0, 4, 5, 1, 2, 3).contiguous()
    elif axis == 1:
        x = x.reshape(B, C0, C2, C3, M_out, L).permute(0, 4, 1, 5, 2, 3).contiguous()
    elif axis == 2:
        x = x.reshape(B, C0, C1, C3, M_out, L).permute(0, 4, 1, 2, 5, 3).contiguous()
    else:
        x = x.reshape(B, C0, C1, C2, M_out, L).permute(0, 4, 1, 2, 3, 5).contiguous()
    return x


class E2E4DBlock(nn.Module):
    """4D Edge-to-Edge block with summed axis-wise 1D convolutions."""

    def __init__(self, in_channels: int, out_channels: int, kernel_radius: int = 1):
        super(E2E4DBlock, self).__init__()
        K = kernel_radius
        kernel_size = 2 * K + 1
        padding = K
        self.conv_axis0 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_axis1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_axis2 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_axis3 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, M_in, C, C, C, C). Out: (B, M_out, C, C, C, C)."""
        B, M, C0, C1, C2, C3 = x.shape
        y0 = _conv1d_along_axis_4d(x, self.conv_axis0, 0)
        y1 = _conv1d_along_axis_4d(x, self.conv_axis1, 1)
        y2 = _conv1d_along_axis_4d(x, self.conv_axis2, 2)
        y3 = _conv1d_along_axis_4d(x, self.conv_axis3, 3)
        return y0 + y1 + y2 + y3


def _e2n_4d_forward(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
                    gamma: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """4D E2N: a_i^(n) = Σ_m Σ_{j,k,l} [ α O_{i,j,k,l} + β O_{j,i,k,l} + γ O_{j,k,i,l} + δ O_{j,k,l,i} ]."""
    # x: (B, M_in, i, j, k, l) = (B, M_in, C, C, C, C)
    B, M_in, C, _, _, _ = x.shape
    M_out = alpha.shape[1]
    # s1: sum_{j,k,l} O_{i,j,k,l} → (B, M_in, C)
    s1 = x.sum(dim=(3, 4, 5))
    # s2: sum_{j,k,l} O_{j,i,k,l} → permute so (B,M,j,i,k,l), sum over j,k,l = dims 2,4,5
    s2 = x.permute(0, 1, 3, 2, 4, 5).contiguous().sum(dim=(2, 4, 5))
    # s3: sum_{j,k,l} O_{j,k,i,l} → (B,M,j,k,i,l), sum over 2,3,5
    s3 = x.permute(0, 1, 2, 4, 3, 5).contiguous().sum(dim=(2, 3, 5))
    # s4: sum_{j,k,l} O_{j,k,l,i} → (B,M,j,k,l,i), sum over 2,3,4
    s4 = x.permute(0, 1, 2, 3, 5, 4).contiguous().sum(dim=(2, 3, 4))
    s = torch.stack([s1, s2, s3, s4], dim=2)   # (B, M_in, 4, C)
    w = torch.stack([alpha, beta, gamma, delta], dim=2)   # (M_in, M_out, 4)
    out = torch.einsum('bmic,mnd->bnic', s, w)
    return out


class E2N4DLayer(nn.Module):
    """4D Edge-to-Node layer with learnable weights for four index placements."""

    def __init__(self, in_channels: int, out_channels: int):
        super(E2N4DLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(in_channels, out_channels) / in_channels)
        self.beta  = nn.Parameter(torch.ones(in_channels, out_channels) / in_channels)
        self.gamma = nn.Parameter(torch.ones(in_channels, out_channels) / in_channels)
        self.delta = nn.Parameter(torch.ones(in_channels, out_channels) / in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, M_in, C, C, C, C) → (B, M_out, C)."""
        return _e2n_4d_forward(x, self.alpha, self.beta, self.gamma, self.delta)


class Brain4DCNN(nn.Module):
    """4th-order tensor encoder: 4D E2E -> 4D E2N -> N2G -> projection head."""

    def __init__(self, example_4d: torch.Tensor, embedding_dim: int = 64,
                 channels: Tuple[int, ...] = (32, 64), kernel_radius: int = 1,
                 dropout_rate: float = 0.5):
        super(Brain4DCNN, self).__init__()
        if example_4d.dim() != 6:
            raise ValueError("Brain4DCNN expects 6D (B, 1, C, C, C, C)")
        _, M0, C, _, _, _ = example_4d.shape
        self.C = C
        self.embedding_dim = embedding_dim
        layers = []
        in_ch = M0
        for out_ch in channels:
            layers.append(E2E4DBlock(in_ch, out_ch, kernel_radius))
            in_ch = out_ch
        self.e2e_layers = nn.ModuleList(layers)
        self.e2n = E2N4DLayer(channels[-1], channels[-1])
        self.fc = nn.Sequential(
            nn.Linear(channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, embedding_dim),
        )

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, C, C, C, C). Out: (B, embedding_dim)."""
        for layer in self.e2e_layers:
            x = F.relu(layer(x))
        # x: (B, M, C, C, C, C)
        x = self.e2n(x)   # (B, M_out, C)
        x = x.mean(dim=2)  # N2G: global average pool over nodes → (B, M_out)
        return self.fc(x)
