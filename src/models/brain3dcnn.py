"""Brain3DCNN for 3rd-order O-information tensors.

Architecture:
- 3D E2E: axis-wise 1D convolutions over three spatial axes.
- 3D E2N: region-centric aggregation from three index placements.
- N2G: permutation-invariant global average pooling to graph embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def _conv1d_along_axis_3d(x: torch.Tensor, conv: nn.Conv1d, axis: int) -> torch.Tensor:
    """Apply 1D convolution along one spatial axis of a 5D tensor."""
    B, M_in, C0, C1, C2 = x.shape
    if axis == 0:
        x = x.permute(0, 3, 4, 1, 2).contiguous()   # (B, C1, C2, M_in, C0)
        batch, L = B * C1 * C2, C0
    elif axis == 1:
        x = x.permute(0, 2, 4, 1, 3).contiguous()   # (B, C0, C2, M_in, C1)
        batch, L = B * C0 * C2, C1
    else:
        x = x.permute(0, 2, 3, 1, 4).contiguous()   # (B, C0, C1, M_in, C2)
        batch, L = B * C0 * C1, C2
    x = x.reshape(batch, M_in, L)
    x = conv(x)
    M_out = x.shape[1]
    if axis == 0:
        x = x.reshape(B, C1, C2, M_out, L).permute(0, 3, 4, 1, 2).contiguous()
    elif axis == 1:
        x = x.reshape(B, C0, C2, M_out, L).permute(0, 3, 1, 4, 2).contiguous()
    else:
        x = x.reshape(B, C0, C1, M_out, L).permute(0, 3, 1, 2, 4).contiguous()
    return x


class E2E3DBlock(nn.Module):
    """3D Edge-to-Edge block with summed axis-wise 1D convolutions."""

    def __init__(self, in_channels: int, out_channels: int, kernel_radius: int = 1):
        super(E2E3DBlock, self).__init__()
        K = kernel_radius
        kernel_size = 2 * K + 1
        padding = K
        self.conv_axis0 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_axis1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_axis2 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, M_in, C, C, C). Out: (B, M_out, C, C, C)."""
        y0 = _conv1d_along_axis_3d(x, self.conv_axis0, 0)
        y1 = _conv1d_along_axis_3d(x, self.conv_axis1, 1)
        y2 = _conv1d_along_axis_3d(x, self.conv_axis2, 2)
        return y0 + y1 + y2


def _e2n_3d_forward(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
                    gamma: torch.Tensor) -> torch.Tensor:
    """3D E2N: a_i^(n) = Σ_m Σ_{j,k} [ α O_{i,j,k} + β O_{j,i,k} + γ O_{j,k,i} ]."""
    # x: (B, M_in, i, j, k)
    B, M_in, C, _, _ = x.shape
    M_out = alpha.shape[1]
    s1 = x.sum(dim=(3, 4))   # sum_{j,k} O_{i,j,k}
    s2 = x.permute(0, 1, 3, 2, 4).contiguous().sum(dim=(2, 4))   # sum_{j,k} O_{j,i,k}
    s3 = x.permute(0, 1, 3, 4, 2).contiguous().sum(dim=(2, 3))   # sum_{j,k} O_{j,k,i}
    s = torch.stack([s1, s2, s3], dim=2)   # (B, M_in, 3, C)
    w = torch.stack([alpha, beta, gamma], dim=2)   # (M_in, M_out, 3)
    out = torch.einsum('bmic,mnd->bnic', s, w)
    return out


class E2N3DLayer(nn.Module):
    """3D Edge-to-Node layer with learnable weights for three index placements."""

    def __init__(self, in_channels: int, out_channels: int):
        super(E2N3DLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(in_channels, out_channels) / in_channels)
        self.beta  = nn.Parameter(torch.ones(in_channels, out_channels) / in_channels)
        self.gamma = nn.Parameter(torch.ones(in_channels, out_channels) / in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, M_in, C, C, C) → (B, M_out, C)."""
        return _e2n_3d_forward(x, self.alpha, self.beta, self.gamma)


class Brain3DCNN(nn.Module):
    """3rd-order tensor encoder: 3D E2E -> 3D E2N -> N2G -> projection head."""

    def __init__(self, example_tensor: torch.Tensor, embedding_dim: int = 64,
                 channels: Tuple[int, ...] = (32, 64), kernel_radius: int = 1,
                 dropout_rate: float = 0.5):
        super(Brain3DCNN, self).__init__()
        if example_tensor.dim() != 5:
            raise ValueError("Brain3DCNN expects 5D (B, 1, C, C, C)")
        _, M0, C, _, _ = example_tensor.shape
        self.C = C
        self.embedding_dim = embedding_dim
        layers = []
        in_ch = M0
        for out_ch in channels:
            layers.append(E2E3DBlock(in_ch, out_ch, kernel_radius))
            in_ch = out_ch
        self.e2e_layers = nn.ModuleList(layers)
        self.e2n = E2N3DLayer(channels[-1], channels[-1])
        self.fc = nn.Sequential(
            nn.Linear(channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, embedding_dim),
        )

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, C, C, C). Out: (B, embedding_dim)."""
        for layer in self.e2e_layers:
            x = F.relu(layer(x))
        x = self.e2n(x)   # (B, M_out, C)
        x = x.mean(dim=2)  # N2G
        return self.fc(x)
