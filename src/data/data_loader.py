"""Data loading for MvHo-IB++ (MvHo-IB-compatible layout)."""

from typing import Dict, List, Optional, Tuple

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .dataset import MultiViewDataset, collate_fn


def _sort_by_sample_id(graph_list: List, tensors: torch.Tensor, labels: torch.Tensor, sample_ids: torch.Tensor):
    idx = sample_ids.argsort()
    tensors = tensors[idx]
    labels = labels[idx]
    sample_ids = sample_ids[idx]

    g_ids = torch.tensor([int(g.sample_id) for g in graph_list], dtype=torch.long)
    g_idx = g_ids.argsort()
    graph_list = [graph_list[i] for i in g_idx.tolist()]
    g_ids = g_ids[g_idx]

    if not torch.equal(g_ids.cpu(), sample_ids.cpu()):
        raise ValueError("sample_ids mismatch between graph and tensor views")
    g_labels = torch.tensor([int(g.y.item()) for g in graph_list], dtype=torch.long)
    if not torch.equal(g_labels.cpu(), labels.cpu()):
        raise ValueError("labels mismatch between graph and tensor views")

    return graph_list, tensors, labels


def load_precomputed_data(config: Dict) -> Tuple:
    """Load graph/x2/x3 tensors and create train/val/test splits.

    Expected dataset config keys:
      x1_path, x2_path, labels_path(optional), x3_path(optional)
    """
    dataset_name = config["dataset_name"]
    ds = config["datasets"][dataset_name]

    x1 = torch.load(ds["x1_path"], map_location="cpu", weights_only=False)
    x2_pack = torch.load(ds["x2_path"], map_location="cpu", weights_only=False)
    x2 = x2_pack["o_matrices"] if isinstance(x2_pack, dict) else x2_pack

    labels = None
    sample_ids = None
    if isinstance(x2_pack, dict):
        labels = x2_pack.get("labels")
        sample_ids = x2_pack.get("sample_ids")

    if labels is None:
        if "labels_path" not in ds:
            raise ValueError("labels not found in x2 pack and labels_path missing")
        labels = torch.load(ds["labels_path"], map_location="cpu")

    if sample_ids is None:
        sample_ids = torch.arange(x2.shape[0], dtype=torch.long)

    x3 = None
    if ds.get("x3_path"):
        x3_pack = torch.load(ds["x3_path"], map_location="cpu", weights_only=False)
        x3 = x3_pack["o4_matrices"] if isinstance(x3_pack, dict) and "o4_matrices" in x3_pack else x3_pack

    x1, x2, labels = _sort_by_sample_id(x1, x2, labels.long(), sample_ids.long())

    indices = list(range(len(x1)))
    test_size = config.get("data_split", {}).get("test_size", 0.1)
    val_size = config.get("data_split", {}).get("val_size", 0.2)
    random_state = config.get("data_split", {}).get("random_state", 42)

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=labels.numpy(),
        random_state=random_state,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        stratify=labels[train_val_idx].numpy(),
        random_state=random_state,
    )

    def pick(idxs):
        g = [x1[i] for i in idxs]
        t2 = x2[idxs]
        t3 = x3[idxs] if x3 is not None else None
        y = labels[idxs]
        return g, t2, t3, y

    return pick(train_idx), pick(val_idx), pick(test_idx)


def create_data_loaders(
    train_split,
    val_split,
    test_split,
    batch_size: int,
    num_workers: int = 0,
):
    """Create train/val/test dataloaders for three-view model."""
    train_ds = MultiViewDataset(*train_split)
    val_ds = MultiViewDataset(*val_split)
    test_ds = MultiViewDataset(*test_split)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    return train_loader, val_loader, test_loader
