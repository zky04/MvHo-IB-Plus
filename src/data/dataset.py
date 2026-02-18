"""Dataset helpers for 3-view training.

View-1: graph data (torch_geometric Data)
View-2: 3D O-info tensor (C,C,C)
View-3: 4D O-info tensor (C,C,C,C), optional
"""

from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class MultiViewDataset(Dataset):
    """Container for graph + 3D + optional 4D tensors and labels."""

    def __init__(
        self,
        graphs: List,
        tensors_3d: torch.Tensor,
        labels: torch.Tensor,
        tensors_4d: Optional[torch.Tensor] = None,
    ) -> None:
        if len(graphs) != tensors_3d.shape[0] or len(graphs) != labels.shape[0]:
            raise ValueError("graphs/tensors_3d/labels sample count mismatch")
        if tensors_4d is not None and tensors_4d.shape[0] != len(graphs):
            raise ValueError("tensors_4d sample count mismatch")
        self.graphs = graphs
        self.tensors_3d = tensors_3d
        self.tensors_4d = tensors_4d
        self.labels = labels.long()

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Tuple:
        x1 = self.graphs[idx]
        x2 = self.tensors_3d[idx]
        y = self.labels[idx]
        x3 = self.tensors_4d[idx] if self.tensors_4d is not None else None
        return x1, x2, x3, y


def collate_fn(batch):
    """Collate graph batch + tensor views.

    Returns:
      graph_batch, x3d(B,C,C,C), x4d_or_none(B,C,C,C,C), labels(B)
    """
    graphs, x3d, x4d, labels = zip(*batch)

    try:
        from torch_geometric.data import Batch
    except ImportError as exc:
        raise ImportError("torch_geometric is required for graph batching") from exc

    graph_batch = Batch.from_data_list(list(graphs))
    x3d_batch = torch.stack(x3d, dim=0)
    if x4d[0] is None:
        x4d_batch = None
    else:
        x4d_batch = torch.stack(x4d, dim=0)
    y_batch = torch.stack(labels, dim=0)
    return graph_batch, x3d_batch, x4d_batch, y_batch
