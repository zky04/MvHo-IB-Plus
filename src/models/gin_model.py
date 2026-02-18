"""GIN (Graph Isomorphism Network) encoder for pairwise connectivity graphs."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, BatchNorm, global_mean_pool


class GINModel(nn.Module):
    """Graph encoder with stacked GINConv, normalization, pooling, and projection."""

    def __init__(self, num_features: int, embedding_dim: int = 64,
                 hidden_dims: list = None, dropout_rate: float = 0.5):
        super(GINModel, self).__init__()
        if hidden_dims is None:
            hidden_dims = [128, 256, 512]
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        dims = [num_features] + hidden_dims
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            )
            self.layers.append(GINConv(mlp))
            self.batch_norms.append(BatchNorm(out_dim))
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dims[-1], embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, batch) -> torch.Tensor:
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        for layer, bn in zip(self.layers, self.batch_norms):
            x = layer(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = global_mean_pool(x, batch_idx)
        return self.output_projection(x)
