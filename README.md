# Graph-based Metric Learning for Scene Understanding

[![Report](https://img.shields.io/badge/Paper-REPORT.md-blue)](docs/REPORT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Informazioni su gruppo e progetto
- **Group ID**: FlyNow
- **Project ID**: 26

## Descrizione del progetto
Il sistema costruisce uno spazio metrico in cui due scene con contenuto semantico
simile finiscono vicine, indipendentemente da colori e texture. Ogni immagine del
dataset GQA viene rappresentata come scene graph, arricchito per inferenza ontologica
tramite Virtuoso e vettorizzato con NOMIC; un graph encoder addestrato in modo
contrastivo (GINEConv o SAGEConv, loss NT-Xent o Triplet) comprime ciascun grafo in un
embedding a 256 dimensioni. Gli embedding vengono indicizzati con FAISS e interrogati
per similarità, con encoder visivi congelati (ResNet50, CLIP) come termine di paragone.

> **Relazione**: per i dettagli teorici, l'analisi delle performance, l'architettura e
> i contributi individuali si rimanda a **[REPORT.md](docs/REPORT.md)**.

## Riproducibilità

### 1. Ambiente e dati

**Prerequisiti**

```bash
git clone https://github.com/flynow-unict/scene-graph-metric-learning.git
cd scene-graph-metric-learning
conda env create -f environment.yml
conda activate dl-scene-retrieval
```

Su GPU, PyTorch va reinstallato con la variante adatta all'hardware: le istruzioni sono
nei commenti in testa a `environment.yml`. Sul cluster GPU l'ambiente arriva già pronto
dal container Apptainer, quindi questo passaggio si salta.

**Dataset**

Immagini, scene graph ed embedding non stanno su GitHub per dimensione. Vanno scaricati
dal [Drive del gruppo](https://drive.google.com/drive/folders/1m_ZYEXbEmvKSD3bNqHsM3GS07wqPDyTT?usp=drive_link)
e scompattati dentro `data/`, mantenendo la struttura descritta in
[`data/README.md`](data/README.md). Chi vuole rigenerare gli scene graph da zero parte
invece dall'istanza Virtuoso, come descritto in
[`src/semantic_web/README.md`](src/semantic_web/README.md).

### 2. Addestramento

Ogni esecuzione è descritta da un file in `experiments/configs/`. I parametri passati
da riga di comando hanno la precedenza su quelli del file.

**Baseline** — solo topologia del grafo, loss NT-Xent

```bash
python -m src.training.train --config experiments/configs/baseline.yaml
```

**Modello principale** — architettura edge-aware, loss Triplet

```bash
python -m src.training.train --config experiments/configs/gine_triplet.yaml
```

Le altre celle della griglia sono `gine_ntxent.yaml` e `sage_triplet.yaml`; le varianti
`gine_ntxent_subset.yaml` e `gine_ntxent_no_enrichment.yaml` isolano rispettivamente
l'effetto della scala del corpus e quello dell'arricchimento ontologico. Sul cluster
SLURM si usano i job in `experiments/jobs/`:

```bash
sbatch experiments/jobs/train_cluster.sh gine triplet
```

### 3. Valutazione

Ground truth di rilevanza, estrazione degli embedding, indice FAISS e metriche:

```bash
python -m src.evaluation.build_relevance --config experiments/configs/relevance.yaml
```

```bash
python -m src.evaluation.extract_gcn_embeddings --config experiments/configs/extract_gine_triplet.yaml
```

```bash
python -m src.evaluation.build_faiss_index && python -m src.evaluation.evaluate
```

Test di robustezza (rimozione progressiva di nodi e archi) ed explainability con
`GNNExplainer`:

```bash
python -m src.evaluation.robustness && python -m src.evaluation.explainability
```

## Risultati

Confronto sul corpus ridotto (15k scene, 771 query), retrieval a K=10. I valori completi
per tutti i modelli, il corpus da 55k, la cross-evaluation zero-shot e i test di
robustezza sono in [`docs/RESULTS.md`](docs/RESULTS.md).

| Modello | Precision@10 | Recall@10 | HitRate@10 |
|---|---:|---:|---:|
| GCN semantica, GINE + Triplet | 15.71 | 39.85 | 74.45 |
| GCN semantica, SAGE + Triplet | 15.16 | 38.29 | 71.98 |
| GCN semantica, GINE + NT-Xent | 14.51 | 36.38 | 70.04 |
| GCN semantica, SAGE + NT-Xent | 13.85 | 35.23 | 68.61 |
| GCN senza arricchimento, GINE + NT-Xent | 11.79 | 30.04 | 60.18 |
| CLIP ViT-B/32 (solo pixel) | 6.86 | 17.59 | 37.35 |
| ResNet50 (solo pixel) | 6.38 | 16.26 | 36.96 |

Sul corpus completo da 55k, con una gallery molto più ampia, lo stesso modello scende a
10.95 / 26.41 / 58.80 e le baseline visive a 5.34 / 12.90 / 32.20 (CLIP).

## Struttura della repository

```
cloud/           backend FastAPI, frontend, worker AI e deploy (Docker, k8s, Terraform, Ansible)
data/            dataset, embedding e checkpoint (non versionati)
docs/            relazione sintetica e risultati degli esperimenti
experiments/     configs/ (YAML), jobs/ (SLURM e script locali), logs/
figures/         grafici prodotti dagli script di valutazione
src/
  datasets/      costruzione scene graph, split, embedding NOMIC, estrattore VLM
  models/        graph encoder e baseline visive (ResNet50, CLIP)
  training/      loop contrastivo e funzioni di loss
  evaluation/    metrica di similarità, indici FAISS, metriche, robustezza
  semantic_web/  client SPARQL, ontologia e demo su Virtuoso
  utils/         percorsi, configurazioni, device e script di servizio
```

---

*Per la dichiarazione dei task individuali e dell'uso dell'AI si rimanda a
[`docs/REPORT.md`](docs/REPORT.md).*
