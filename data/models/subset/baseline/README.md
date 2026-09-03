# Embedding delle baseline visive

Termine di paragone per misurare quanto il graph encoder guadagni rispetto alle
architetture puramente visive. Questi vettori descrivono ogni immagine partendo dai soli
pixel, senza passare dallo scene graph.

## Modelli

- **ResNet50** — CNN preaddestrata su ImageNet, vettori a 2048 dimensioni.
- **CLIP ViT-B/32** — Vision Transformer preaddestrato su coppie immagine-testo, vettori
  a 512 dimensioni in uno spazio multimodale.

Entrambi gli encoder sono congelati: nessun fine-tuning sui dati del progetto.

## Formato dei file

Ogni file `.pt` è un dizionario salvato con `torch.save`.

```python
{
  "embeddings": Tensor[N, D],   # vettori L2-normalizzati (D = 2048 o 512)
  "image_ids":  [int, ...],     # ID originali di GQA
  "dim":        D,
  "split":      "test_gallery" | "test_queries",
  "encoder":    "resnet" | "clip"
}
```

## Rigenerazione

```bash
python -m src.evaluation.extract_baseline_embeddings --encoder resnet --split test_gallery
```
