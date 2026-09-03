"""Graph encoder: da un grafo a un embedding a dimensione fissa."""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINEConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool


_CONV = {"gcn": GCNConv, "gat": GATConv, "sage": SAGEConv, "gine": GINEConv}
_POOL = {"mean": global_mean_pool, "add": global_add_pool, "max": global_max_pool}


class GraphEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 256,
        out_dim: int = 256,
        num_layers: int = 3,
        conv_type: str = "sage",
        pool: str = "mean",
        dropout: float = 0.3,
        normalize_output: bool = True,
        edge_dim: int = 768,
    ):
        super().__init__()
        if conv_type not in _CONV:
            raise ValueError(f"conv_type deve essere uno di {list(_CONV)}")
        if pool not in _POOL:
            raise ValueError(f"pool deve essere uno di {list(_POOL)}")

        Conv = _CONV[conv_type]
        self.pool = _POOL[pool]
        self.dropout = dropout
        self.normalize_output = normalize_output
        # 'gine' e' l'unica conv edge-aware: usa gli embedding NOMIC delle relazioni
        # (edge_attr) nel message passing; le altre vedono solo la topologia.
        self.edge_aware = conv_type == "gine"
        self.edge_dim = edge_dim

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            if self.edge_aware:
                mlp = nn.Sequential(
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.ReLU(inplace=True),
                    nn.Linear(dims[i + 1], dims[i + 1]),
                )
                self.convs.append(Conv(mlp, edge_dim=edge_dim))
            else:
                self.convs.append(Conv(dims[i], dims[i + 1]))
            self.norms.append(nn.BatchNorm1d(dims[i + 1]))

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        if self.edge_aware and edge_attr is None:
            # grafi senza edge_attr (mock, o dati vecchi): relazione "muta" a zero
            edge_attr = x.new_zeros(edge_index.size(1), self.edge_dim)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr) if self.edge_aware else conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        g = self.pool(x, batch)
        g = self.proj(g)
        if self.normalize_output:
            g = F.normalize(g, p=2, dim=-1)
        return g
