"""Estrazione embedding dei grafi col GraphEncoder addestrato (per il retrieval).

Questo script è speculare a extract_baseline_embeddings.py, ma opera sugli scene graph 
anziché sulle immagini. L'output è salvato nello stesso formato dei baseline per
garantire la compatibilità con un'unica pipeline di valutazione (es. tramite FAISS).

    python -m src.evaluation.extract_gcn_embeddings --config experiments/configs/extract_gine_triplet.yaml
    python -m src.evaluation.extract_gcn_embeddings --split test_queries --ckpt data/models/fullset/checkpoints/gcn_encoder_gine_triplet.pth
"""
import argparse
import os

import torch

from src.datasets.graph_dataset import load_dataset
from src.models.graph_encoder import GraphEncoder
from src.utils.config import parse_with_config
from src.utils.device import get_device
from src.utils import paths


# Split predefiniti del corpus completo arricchito. Per gli altri corpus si passa
# --graphs esplicitamente, oppure si usa un file di configurazione.
SPLITS = {
    name: str(paths.scene_graphs() / f"{name}_scene_graphs.pt")
    for name in ("test_gallery", "test_queries", "val")
}


def build_model_from_ckpt(ckpt, device):
    cfg = ckpt.get("config", {})
    model = GraphEncoder(
        in_dim=cfg.get("in_dim", 768),
        hidden_dim=cfg.get("hidden_dim", 256),
        out_dim=cfg.get("out_dim", 256),
        num_layers=cfg.get("num_layers", 3),
        conv_type=cfg.get("conv", "sage"),
        pool=cfg.get("pool", "mean"),
        dropout=cfg.get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["model_state"])
    return model.eval().to(device)


@torch.no_grad()
def extract(args):
    from torch_geometric.data import Batch

    device = get_device()

    graphs_path = args.graphs or SPLITS[args.split]
    if not os.path.exists(graphs_path):
        raise FileNotFoundError(f"File grafi non trovato: {graphs_path}")
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint non trovato: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_model_from_ckpt(ckpt, device)

    graphs = load_dataset(graphs_path)
    if args.limit:
        graphs = graphs[:args.limit]
    print(f"[dataset] {graphs_path}: {len(graphs)} grafi | out_dim={model.proj[-1].out_features}")

    all_emb, all_ids = [], []
    for i in range(0, len(graphs), args.batch_size):
        chunk = graphs[i:i + args.batch_size]
        batch = Batch.from_data_list(chunk).to(device)
        z = model(batch.x, batch.edge_index, batch.batch,
                  getattr(batch, "edge_attr", None)).cpu()
        all_emb.append(z)
        all_ids.extend(int(g.image_id) for g in chunk)

    embeddings = torch.cat(all_emb, dim=0)
    out = {"embeddings": embeddings, "image_ids": all_ids,
           "dim": embeddings.size(1), "split": args.split, "encoder": "gcn"}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(out, args.out)
    print(f"[done] {embeddings.size(0)} embedding ({embeddings.size(1)}-dim) -> {args.out}")


def build_parser():
    p = argparse.ArgumentParser(description="Estrazione embedding dei grafi (GCN)")
    p.add_argument("--split", choices=list(SPLITS), default="test_gallery")
    p.add_argument("--graphs", type=str, default=None)
    p.add_argument("--ckpt", type=str,
                   default=str(paths.checkpoints() / "gcn_encoder.pth"))
    p.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=str, default=None)
    return p


if __name__ == "__main__":
    args = parse_with_config(build_parser())
    if args.out is None:
        args.out = str(paths.embeddings() / f"gcn_{args.split}.pt")
    extract(args)
