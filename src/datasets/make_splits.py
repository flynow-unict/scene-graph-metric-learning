"""Genera gli split train/val/test dal dataset completo, con seed fisso.

    python -m src.datasets.make_splits
"""
import argparse
import os
import random

import torch

RATIOS = {"train": 0.65, "val": 0.10, "test_gallery": 0.20, "test_queries": 0.05}


def make_splits(input_path: str, out_dir: str, seed: int = 42):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset non trovato: {input_path}")

    dataset = torch.load(input_path, weights_only=False)
    n = len(dataset)
    print(f"Caricati {n} grafi da {input_path}")

    idx = list(range(n))
    random.seed(seed)
    random.shuffle(idx)
    data = [dataset[i] for i in idx]

    n_train = int(RATIOS["train"] * n)
    n_val = int(RATIOS["val"] * n)
    n_gal = int(RATIOS["test_gallery"] * n)

    splits = {
        "train": data[:n_train],
        "val": data[n_train:n_train + n_val],
        "test_gallery": data[n_train + n_val:n_train + n_val + n_gal],
        "test_queries": data[n_train + n_val + n_gal:],
    }

    os.makedirs(out_dir, exist_ok=True)
    for name, part in splits.items():
        path = os.path.join(out_dir, f"{name}_embedded.pt")
        torch.save(part, path)
        print(f"  {name:13s} {len(part):6d} grafi -> {path}")


def build_parser():
    p = argparse.ArgumentParser(description="Split del dataset (seed fisso)")
    p.add_argument("--input", default="data/embedding/Rel/dataset_gqa_embedded.pt")
    p.add_argument("--out-dir", dest="out_dir", default="data/embedding/Rel")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    a = build_parser().parse_args()
    make_splits(a.input, a.out_dir, a.seed)
