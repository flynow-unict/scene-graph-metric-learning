"""Utility per i dati: dataset finto, caricamento, augmentation dei grafi."""
from typing import List

import torch
from torch_geometric.data import Data


def make_mock_dataset(num_graphs: int = 512, feat_dim: int = 768,
                      min_nodes: int = 3, max_nodes: int = 20,
                      seed: int = 42) -> List[Data]:
    """Grafi casuali per testare la rete senza i dati reali."""
    g = torch.Generator().manual_seed(seed)
    dataset = []
    for i in range(num_graphs):
        n = int(torch.randint(min_nodes, max_nodes + 1, (1,), generator=g))
        x = torch.randn(n, feat_dim, generator=g)
        e = max(1, int(n * 1.5))
        edge_index = torch.randint(0, n, (2, e), generator=g)
        data = Data(x=x, edge_index=edge_index)
        data.image_id = i
        dataset.append(data)
    return dataset


def load_dataset(path: str) -> List[Data]:
    dataset = torch.load(path, weights_only=False)
    if not isinstance(dataset, (list, tuple)):
        raise TypeError(f"Atteso una lista di Data, ottenuto {type(dataset)}")
    return list(dataset)


def augment_graph(data: Data, node_drop: float = 0.1, edge_drop: float = 0.2,
                  feat_mask: float = 0.1) -> data:
    """Vista aumentata del grafo: drop di nodi/archi e masking delle feature."""
    x, edge_index = data.x, data.edge_index
    n = x.size(0)

    keep_mask = torch.rand(n) > node_drop
    if keep_mask.sum() < 2:
        keep_mask = torch.ones(n, dtype=torch.bool)
    keep_idx = keep_mask.nonzero(as_tuple=True)[0]
    remap = -torch.ones(n, dtype=torch.long)
    remap[keep_idx] = torch.arange(keep_idx.size(0))

    new_x = x[keep_idx].clone()
    if feat_mask > 0:
        fm = torch.rand_like(new_x) > feat_mask
        new_x = new_x * fm

    edge_attr = getattr(data, "edge_attr", None)
    if edge_index.size(1) > 0:
        src, dst = edge_index
        valid = keep_mask[src] & keep_mask[dst]
        valid &= torch.rand(edge_index.size(1)) > edge_drop
        new_edge = torch.stack([remap[src[valid]], remap[dst[valid]]], dim=0)
        new_edge_attr = edge_attr[valid].clone() if edge_attr is not None else None
    else:
        new_edge = torch.empty((2, 0), dtype=torch.long)
        new_edge_attr = edge_attr[:0].clone() if edge_attr is not None else None

    out = Data(x=new_x, edge_index=new_edge)
    if new_edge_attr is not None:
        out.edge_attr = new_edge_attr
    if hasattr(data, "image_id"):
        out.image_id = data.image_id
    return out
