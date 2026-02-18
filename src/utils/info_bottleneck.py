"""Information bottleneck utilities (matrix-based Renyi entropy)."""

from ..models.fusion_model import renyi_entropy, mutual_information_loss, three_view_ib_loss

__all__ = ["renyi_entropy", "mutual_information_loss", "three_view_ib_loss"]
