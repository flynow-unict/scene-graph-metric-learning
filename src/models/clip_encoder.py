"""Encoder per estrarre le caratteristiche visive delle immagini tramite CLIP."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _CLIPPreprocess:
    """Gestisce la preparazione delle immagini prima di passarle a CLIP.

    È scritta come classe separata per permettere al DataLoader di usare più processi 
    in parallelo (num_workers > 0) per caricare i dati più velocemente. Le normali
    funzioni create al volo non possono essere condivise tra processi diversi, 
    mentre questa classe sì.
    """

    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, pil_image):
        out = self.image_processor(images=pil_image, return_tensors="pt")
        return out["pixel_values"][0]


class CLIPImageEncoder(nn.Module):
    """Modello per estrarre le informazioni dalle immagini usando CLIP.
    
    Prende in input le immagini e restituisce dei vettori numerici (embedding) 
    che ne rappresentano il contenuto visivo.
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 normalize_output: bool = True):
        super().__init__()
        from transformers import CLIPModel, CLIPImageProcessor

        self.model = CLIPModel.from_pretrained(model_name)
        self.model.eval()
        self.normalize_output = normalize_output
        self.feature_dim = self.model.config.projection_dim  # 512

        image_processor = CLIPImageProcessor.from_pretrained(model_name)
        self.preprocess = _CLIPPreprocess(image_processor)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        vision_out = self.model.vision_model(pixel_values=images)
        pooled = vision_out.pooler_output
        feats = self.model.visual_projection(pooled)
        if self.normalize_output:
            feats = F.normalize(feats, p=2, dim=-1)
        return feats
