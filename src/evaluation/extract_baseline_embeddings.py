"""Estrazione embedding immagini (ResNet o CLIP) per il retrieval.

    python -m src.evaluation.extract_baseline_embeddings --encoder resnet --split test_gallery
    python -m src.evaluation.extract_baseline_embeddings --encoder clip --split test_queries
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from src.datasets.image_dataset import GQAImageDataset
from src.models.clip_encoder import CLIPImageEncoder
from src.models.resnet_baseline import ResNetBaseline
from src.utils.config import parse_with_config
from src.utils.device import get_device
from src.utils import paths


def build_encoder(name: str, out_dim=None):
    if name == "resnet":
        return ResNetBaseline(out_dim=out_dim, pretrained=True, normalize_output=True)
    if name == "clip":
        return CLIPImageEncoder(normalize_output=True)
    raise ValueError(f"encoder sconosciuto: {name}")


# I grafi servono solo a sapere quali image_id compongono lo split: gli encoder
# visivi lavorano sui pixel.
SPLITS = {
    name: str(paths.scene_graphs() / f"{name}_scene_graphs.pt")
    for name in ("test_gallery", "test_queries", "val")
}


@torch.no_grad()
def extract(args):
    device = get_device()

    graphs_path = args.graphs or SPLITS[args.split]
    if not os.path.exists(graphs_path):
        raise FileNotFoundError(f"File grafi non trovato: {graphs_path}")

    model = build_encoder(args.encoder, out_dim=args.out_dim)
    model.eval().to(device)
    print(f"[encoder] {args.encoder} | dim = {getattr(model, 'feature_dim', '?')}")

    dataset = GQAImageDataset(graphs_path, args.images_dir, transform=model.preprocess)
    if len(dataset) == 0:
        raise RuntimeError("Nessuna immagine presente su disco per questo split.")
    if args.limit:
        from torch.utils.data import Subset
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
        print(f"[dataset] --limit attivo: {len(dataset)} immagini")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)

    all_emb, all_ids = [], []
    for imgs, ids in loader:
        emb = model(imgs.to(device)).cpu()
        all_emb.append(emb)
        all_ids.extend(int(i) for i in ids)

    embeddings = torch.cat(all_emb, dim=0)
    out = {"embeddings": embeddings, "image_ids": all_ids,
           "dim": embeddings.size(1), "split": args.split, "encoder": args.encoder}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(out, args.out)
    print(f"[done] {embeddings.size(0)} embedding ({embeddings.size(1)}-dim) -> {args.out}")


def build_parser():
    p = argparse.ArgumentParser(description="Estrazione embedding immagini")
    p.add_argument("--encoder", choices=["resnet", "clip"], default="resnet")
    p.add_argument("--split", choices=list(SPLITS), default="test_gallery")
    p.add_argument("--graphs", type=str, default=None)
    p.add_argument("--images-dir", dest="images_dir", default=str(paths.IMAGES_DIR))
    p.add_argument("--out-dim", dest="out_dim", type=int, default=None)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=str, default=None)
    return p


if __name__ == "__main__":
    args = parse_with_config(build_parser())
    if args.out is None:
        args.out = str(paths.MODELS_DIR / "fullset" / "vision_baselines"
                       / f"{args.encoder}_{args.split}.pt")
    extract(args)
