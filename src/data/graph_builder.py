"""Build 2D view graph from ROI time series using matrix-based Renyi MI.

Methodology alignment:
- V1(i,j) = I_alpha(X_i; X_j)
- keep top-ratio edges (default 30%) by MI magnitude.
"""

from typing import Optional, Tuple

import torch


def _gram_1d(x: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
    x = x.view(-1, 1)
    d2 = (x - x.t()).pow(2)
    if sigma is None:
        sigma = torch.sqrt(torch.median(d2[d2 > 0])) if torch.any(d2 > 0) else torch.tensor(1.0, device=x.device)
    sigma = torch.clamp(sigma, min=1e-6)
    k = torch.exp(-d2 / (2.0 * sigma * sigma))
    k = k / (torch.trace(k) + 1e-12)
    return k


def _renyi_entropy_from_gram(k: torch.Tensor, alpha: float = 1.01) -> torch.Tensor:
    evals = torch.linalg.eigvalsh(k).clamp(min=1e-12)
    return torch.log2(torch.sum(evals.pow(alpha))) / (1.0 - alpha)


def renyi_mutual_information(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.01) -> torch.Tensor:
    kx = _gram_1d(x)
    ky = _gram_1d(y)
    h_x = _renyi_entropy_from_gram(kx, alpha=alpha)
    h_y = _renyi_entropy_from_gram(ky, alpha=alpha)
    kxy = (kx * ky)
    kxy = kxy / (torch.trace(kxy) + 1e-12)
    h_xy = _renyi_entropy_from_gram(kxy, alpha=alpha)
    return h_x + h_y - h_xy


def build_renyi_connectivity(time_courses: torch.Tensor, alpha: float = 1.01) -> torch.Tensor:
    """Build dense V1 matrix from (T,C) time courses."""
    if time_courses.dim() != 2:
        raise ValueError("time_courses must be (T, C)")
    _, c = time_courses.shape
    v1 = torch.zeros(c, c, dtype=torch.float32)
    for i in range(c):
        v1[i, i] = 0.0
        for j in range(i + 1, c):
            mi = renyi_mutual_information(time_courses[:, i], time_courses[:, j], alpha=alpha)
            v = float(mi.item())
            v1[i, j] = v
            v1[j, i] = v
    return v1


def sparsify_top_ratio(matrix: torch.Tensor, top_ratio: float = 0.3) -> torch.Tensor:
    """Keep top-k off-diagonal entries by absolute value."""
    if not (0.0 < top_ratio <= 1.0):
        raise ValueError("top_ratio must be in (0,1]")
    m = matrix.clone()
    c = m.shape[0]
    triu_i, triu_j = torch.triu_indices(c, c, offset=1)
    vals = m[triu_i, triu_j].abs()
    k = max(1, int(vals.numel() * top_ratio))
    topk = torch.topk(vals, k).values.min()
    mask_upper = vals >= topk

    out = torch.zeros_like(m)
    keep_i = triu_i[mask_upper]
    keep_j = triu_j[mask_upper]
    out[keep_i, keep_j] = m[keep_i, keep_j]
    out[keep_j, keep_i] = m[keep_j, keep_i]
    return out


def build_renyi_graph_from_timeseries(
    time_courses: torch.Tensor,
    label: int,
    sample_id: int,
    alpha: float = 1.01,
    top_ratio: float = 0.3,
) -> Tuple[torch.Tensor, object]:
    """Build sparse MI matrix and torch_geometric Data graph object."""
    v1 = build_renyi_connectivity(time_courses, alpha=alpha)
    v1_sparse = sparsify_top_ratio(v1, top_ratio=top_ratio)

    edge_index = (v1_sparse != 0).nonzero(as_tuple=False).t().contiguous()
    edge_weight = v1_sparse[edge_index[0], edge_index[1]].float()
    x = torch.ones((v1_sparse.shape[0], 1), dtype=torch.float32)

    try:
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError("torch_geometric is required for view-1 graph construction") from exc

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=torch.tensor([label], dtype=torch.long))
    graph.sample_id = int(sample_id)
    return v1_sparse, graph
