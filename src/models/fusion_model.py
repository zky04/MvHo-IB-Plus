"""Three-view fusion model with information bottleneck regularization.

View 1: graph -> GIN -> Z_gin
View 2: 3D tensor -> Brain3DCNN -> Z_3d
View 3: 4D tensor -> Brain4DCNN -> Z_4d

Fusion path: concat(Z_gin, Z_3d, Z_4d) -> MLP -> logits.
Training objective: L_cls + three_view_ib_loss(z_gin, z_3d, z_4d, ...).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


# ---------- Information Bottleneck Terms (Renyi entropy + pairwise MI) ----------


def _gram_matrix(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """K[i,j] = exp(-||x_i - x_j||^2 / sigma)。"""
    x = x.view(x.shape[0], -1)
    instances_norm = torch.sum(x ** 2, dim=-1).reshape((-1, 1))
    pairwise_distances = -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()
    return torch.exp(-pairwise_distances / sigma)


def renyi_entropy(x: torch.Tensor, sigma: float = 5.0, alpha: float = 1.01, epsilon: float = 1e-3) -> torch.Tensor:
    """Matrix-based Renyi alpha entropy: H_alpha(X) = 1/(1-alpha) * log2(sum lambda_i^alpha)."""
    x = x.double()
    k = _gram_matrix(x, sigma)
    trace_k = torch.trace(k)
    if trace_k == 0:
        trace_k = 1e-10
    k = k / trace_k
    k = k + epsilon * torch.eye(k.size(0), dtype=torch.double, device=k.device)
    try:
        eigenvalues = torch.linalg.eigh(k, UPLO='U')[0]
        eigenvalues = torch.abs(eigenvalues)
    except RuntimeError:
        return torch.tensor(0.0, device=x.device, dtype=torch.float32)
    sum_pow = torch.sum(eigenvalues ** alpha)
    entropy = (1.0 / (1.0 - alpha)) * torch.log2(sum_pow + 1e-10)
    return entropy.float()


def mutual_information_loss(z1: torch.Tensor, z2: torch.Tensor,
                            sigma: float = 5.0, alpha: float = 1.01) -> torch.Tensor:
    """Two-view MI loss (negative MI for minimization). MI = H(Z1) + H(Z2) - H(Z1,Z2)."""
    h_z1 = renyi_entropy(z1, sigma, alpha)
    h_z2 = renyi_entropy(z2, sigma, alpha)
    z_joint = torch.cat([z1, z2], dim=1)
    h_z1_z2 = renyi_entropy(z_joint, sigma, alpha)
    return -(h_z1 + h_z2 - h_z1_z2)


def three_view_ib_loss(z_gin: Optional[torch.Tensor], z_3d: Optional[torch.Tensor], z_4d: Optional[torch.Tensor],
                       beta_gin: float = 0.01, beta_3d: float = 0.01, beta_4d: float = 0.01,
                       beta_mutual: float = 0.01, sigma: float = 5.0, alpha: float = 1.01) -> torch.Tensor:
    """Three-view IB term: weighted entropies plus weighted pairwise MI penalties."""
    device = None
    for z in [z_gin, z_3d, z_4d]:
        if z is not None:
            device = z.device
            break
    loss = torch.tensor(0.0, device=device if device is not None else torch.device('cpu'))
    if z_gin is not None:
        loss = loss + beta_gin * renyi_entropy(z_gin, sigma, alpha)
    if z_3d is not None:
        loss = loss + beta_3d * renyi_entropy(z_3d, sigma, alpha)
    if z_4d is not None:
        loss = loss + beta_4d * renyi_entropy(z_4d, sigma, alpha)
    if z_gin is not None and z_3d is not None:
        loss = loss + beta_mutual * mutual_information_loss(z_gin, z_3d, sigma, alpha)
    if z_gin is not None and z_4d is not None:
        loss = loss + beta_mutual * mutual_information_loss(z_gin, z_4d, sigma, alpha)
    if z_3d is not None and z_4d is not None:
        loss = loss + beta_mutual * mutual_information_loss(z_3d, z_4d, sigma, alpha)
    return loss


# ---------- Fusion Model ----------


class FusionModel(nn.Module):
    """Multi-view feature fusion model（3-view：GIN + Brain3DCNN + Brain4DCNN）。"""

    def __init__(self,
                 input_size: int,
                 num_classes: int,
                 dropout_rate: float = 0.5) -> None:
        super(FusionModel, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.fusion_network = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.classifier = nn.Linear(input_size // 4, num_classes)

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, *embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            *embeddings: 1-3 view embeddings, e.g. (z_gin, z_3d, z_4d); None entries are ignored.

        Returns:
            (fused_features, logits)
        """
        valid_embeddings = [emb for emb in embeddings if emb is not None]
        if not valid_embeddings:
            raise ValueError("At least one valid embedding input is required")

        if len(valid_embeddings) == 1:
            fused_features = valid_embeddings[0]
        else:
            fused_features = torch.cat(valid_embeddings, dim=1)

        fused_features = self.fusion_network(fused_features)
        logits = self.classifier(fused_features)
        return fused_features, logits

    def get_fusion_features(self, *embeddings: torch.Tensor) -> torch.Tensor:
        """Return fused features before the classifier head."""
        valid_embeddings = [emb for emb in embeddings if emb is not None]
        if not valid_embeddings:
            raise ValueError("At least one valid feature embedding must be provided")
        if len(valid_embeddings) == 1:
            return valid_embeddings[0]
        return torch.cat(valid_embeddings, dim=1)
