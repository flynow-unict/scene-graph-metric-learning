"""Dataset immagini allineato ai grafi tramite image_id."""
import os
from typing import Callable, List

import torch
from PIL import Image
from torch.utils.data import Dataset


class GQAImageDataset(Dataset):
    def __init__(self, graphs_path: str, images_dir: str, transform: Callable):
        graphs = torch.load(graphs_path, weights_only=False)
        self.images_dir = images_dir
        self.transform = transform

        self.image_ids: List[int] = []
        missing = 0
        for g in graphs:
            iid = int(g.image_id)
            if os.path.exists(self._path(iid)):
                self.image_ids.append(iid)
            else:
                missing += 1

        self.total = len(graphs)
        self.missing = missing
        print(f"[dataset] {graphs_path}: {len(self.image_ids)}/{self.total} "
              f"immagini presenti ({missing} mancanti)")

    def _path(self, image_id: int) -> str:
        return os.path.join(self.images_dir, f"{image_id}.jpg")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        iid = self.image_ids[idx]
        img = Image.open(self._path(iid)).convert("RGB")
        return self.transform(img), iid
