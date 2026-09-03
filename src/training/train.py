"""Training contrastivo del graph encoder (NT-Xent o Triplet).

    python -m src.training.train --config experiments/configs/baseline.yaml
    python -m src.training.train --mock --epochs 2
"""
import argparse
import os
import random

import torch

from src.datasets.graph_dataset import make_mock_dataset, load_dataset, augment_graph
from src.evaluation.scene_similarity import (
    graph_signature, mine_positives, compute_scene_embeddings, compute_idf)
from src.models.graph_encoder import GraphEncoder
from src.training.losses import NTXentLoss, batch_hard_triplet_loss
from src.utils.config import parse_with_config
from src.utils.device import get_device
from src.utils import paths


def iter_batches(n, batch_size, shuffle=True):
    """Genera batch di indici (necessario per recuperare agevolmente i positivi)."""
    idx = list(range(n))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        yield idx[i:i + batch_size]


def to_batch(graph_list, device):
    """Aggrega un batch mantenendo esclusivamente gli attributi necessari al modello
    (x, edge_index, edge_attr): previene errori nella funzione di collate di PyG 
    dovuti ad attributi non tensoriali (node_text, image_id...) e permette di unire 
    grafi reali e aumentati nello stesso batch."""
    from torch_geometric.data import Batch, Data
    slim = []
    for g in graph_list:
        d = Data(x=g.x, edge_index=g.edge_index)
        if getattr(g, "edge_attr", None) is not None:
            d.edge_attr = g.edge_attr
        slim.append(d)
    return Batch.from_data_list(slim).to(device)


def positive_graphs(batch_idx, dataset, pos_index):
    """Restituisce il grafo positivo per ciascuna ancora.
    Viene selezionato casualmente uno dei top-k trovati per incoraggiare il modello 
    ad apprendere le caratteristiche dello scenario piuttosto che memorizzare la 
    singola coppia. Se non ci sono positivi affidabili, si ripiega su una vista 
    aumentata dell'ancora originale."""
    out = []
    for j in batch_idx:
        cands = pos_index[j] if pos_index is not None else []
        if cands:
            pj, _ = random.choice(cands)
            out.append(dataset[pj])
        else:
            out.append(augment_graph(dataset[j]))
    return out


@torch.no_grad()
def collapse_metrics(z):
    """Indicatori di collasso: coseno medio tra elementi DIVERSI del batch
    (verso 1 = collasso) e std media per-dimensione (verso 0 = collasso)."""
    B = z.size(0)
    sim = z @ z.t()
    off = sim[~torch.eye(B, dtype=torch.bool, device=z.device)]
    return off.mean().item(), z.std(0).mean().item()


def train(args):
    device = get_device()

    if args.mock or not args.data:
        print("[data] dataset finto (mock).")
        dataset = make_mock_dataset(num_graphs=args.mock_size, feat_dim=args.in_dim)
    else:
        print(f"[data] carico {args.data}")
        dataset = load_dataset(args.data)
    print(f"[data] {len(dataset)} grafi | feat_dim={dataset[0].x.size(1)}")

    model = GraphEncoder(
        in_dim=args.in_dim, hidden_dim=args.hidden_dim, out_dim=args.out_dim,
        num_layers=args.num_layers, conv_type=args.conv, pool=args.pool,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Cosine annealing: il learning rate decresce verso lo 0 nel corso delle epoche.
    # Nelle fasi finali i gradienti sono molto piccoli, consentendo alla loss di 
    # convergere stabilmente al minimo senza oscillazioni.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ntxent = NTXentLoss(temperature=args.temperature)

    # Selezione positivi: 
    # 'augment' = due viste perturbate dello stesso grafo (self-supervised).
    # 'mixed' = scena diversa ma semanticamente simile (metric learning).
    # Quest'ultima modalità richiede node_text/edge_text (assenti nel dataset mock).
    pos_index = None
    if args.positives == "mixed":
        if not hasattr(dataset[0], "node_text"):
            raise SystemExit("--positives mixed richiede grafi con node_text/edge_text "
                             "(usa i dati reali, non --mock).")
        print("[positives] mining scene simili (mode=mixed, top-k + veto semantico)...")
        sigs = [graph_signature(g) for g in dataset]
        weights = compute_idf(sigs)
        embs = compute_scene_embeddings(dataset, weights=weights[0])
        pos_index = mine_positives(dataset, mode="mixed", min_sim=args.min_sim,
                                   signatures=sigs, weights=weights, k=args.pos_k,
                                   embeddings=embs, sem_veto=args.sem_veto)
        found = sum(1 for lst in pos_index if lst)
        avg_k = sum(len(lst) for lst in pos_index) / max(1, found)
        print(f"[positives] {found}/{len(dataset)} ancore con positivi >= {args.min_sim} "
              f"(media {avg_k:.1f} positivi/ancora; le altre ricadono sull'augmentation)")

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="fly-now-gcn", config=vars(args))
        wandb.watch(model)

    print(f"[train] loss={args.loss} | epochs={args.epochs} | batch={args.batch_size}")
    model.train()
    for epoch in range(1, args.epochs + 1):
        total, n_batches = 0.0, 0
        z_mon = None
        for batch_idx in iter_batches(len(dataset), args.batch_size, shuffle=True):
            optimizer.zero_grad()
            batch_list = [dataset[j] for j in batch_idx]

            if args.positives == "augment":
                # due viste aumentate (ntxent) o ancora+vista (triplet)
                anchor_list = batch_list
                pos_list = [augment_graph(dataset[j]) for j in batch_idx]
                view1_list = [augment_graph(g) for g in batch_list] if args.loss == "ntxent" else batch_list
            else:
                # positivo = scena simile minata (fallback augmentation dentro positive_graphs)
                anchor_list = batch_list
                view1_list = batch_list
                pos_list = positive_graphs(batch_idx, dataset, pos_index)

            if args.loss == "ntxent":
                v1 = to_batch(view1_list, device)
                v2 = to_batch(pos_list, device)
                z1 = model(v1.x, v1.edge_index, v1.batch, getattr(v1, "edge_attr", None))
                z2 = model(v2.x, v2.edge_index, v2.batch, getattr(v2, "edge_attr", None))
                loss = ntxent(z1, z2)
                z_mon = z1.detach()
            else:
                anchor = to_batch(anchor_list, device)
                pos = to_batch(pos_list, device)
                za = model(anchor.x, anchor.edge_index, anchor.batch, getattr(anchor, "edge_attr", None))
                zp = model(pos.x, pos.edge_index, pos.batch, getattr(pos, "edge_attr", None))
                loss = batch_hard_triplet_loss(za, zp, margin=args.margin,
                                               mining=args.triplet_mining)
                z_mon = za.detach()

            loss.backward()
            optimizer.step()
            total += loss.item()
            n_batches += 1

        avg = total / max(1, n_batches)
        off_cos, dim_std = collapse_metrics(z_mon)
        lr_now = scheduler.get_last_lr()[0]
        flag = "  <-- COLLASSO" if off_cos > 0.9 else ""
        print(f"  epoch {epoch:3d}/{args.epochs} | loss={avg:.4f} | "
              f"off_cos={off_cos:+.3f} dim_std={dim_std:.4f} | lr={lr_now:.2e}{flag}")
        if run:
            run.log({"epoch": epoch, "loss": avg, "lr": lr_now,
                     "off_cos": off_cos, "dim_std": dim_std})
        scheduler.step()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": vars(args)}, args.out)
    print(f"[done] pesi salvati in {args.out}")
    if run:
        run.finish()


def build_parser():
    p = argparse.ArgumentParser(description="Training contrastivo del graph encoder")
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--mock-size", dest="mock_size", type=int, default=512)
    p.add_argument("--in-dim", dest="in_dim", type=int, default=768)
    p.add_argument("--hidden-dim", dest="hidden_dim", type=int, default=256)
    p.add_argument("--out-dim", dest="out_dim", type=int, default=256)
    p.add_argument("--num-layers", dest="num_layers", type=int, default=3)
    p.add_argument("--conv", choices=["gcn", "gat", "sage", "gine"], default="sage",
                   help="gine = edge-aware: usa gli embedding NOMIC delle relazioni")
    p.add_argument("--pool", choices=["mean", "add", "max"], default="mean")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--loss", choices=["ntxent", "triplet"], default="ntxent")
    p.add_argument("--positives", choices=["augment", "mixed"], default="mixed",
                   help="augment=viste aumentate (self-supervised); "
                        "mixed=scena simile minata con scene_similarity (metric learning)")
    p.add_argument("--min-sim", dest="min_sim", type=float, default=0.15,
                   help="soglia minima di similarita' per accettare un positivo minato")
    p.add_argument("--pos-k", dest="pos_k", type=int, default=5,
                   help="quanti positivi tenere per ancora (a ogni step se ne campiona uno)")
    p.add_argument("--sem-veto", dest="sem_veto", type=float, default=0.55,
                   help="coseno NOMIC minimo (embedding pesati IDF e centrati) per "
                        "accettare un positivo con Jaccard debole (<0.30); i match con "
                        "Jaccard forte non passano dal veto. Scelto su curva "
                        "precision/coverage con CLIP come giudice visivo esterno")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--triplet-mining", dest="triplet_mining",
                   choices=["semi", "hard"], default="semi")
    p.add_argument("--out", type=str,
                   default=str(paths.checkpoints() / "gcn_encoder.pth"))
    p.add_argument("--wandb", action="store_true")
    return p


if __name__ == "__main__":
    train(parse_with_config(build_parser()))
