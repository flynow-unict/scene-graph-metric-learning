# data

Dataset, embedding e checkpoint. Il contenuto non è versionato per dimensione: git
traccia soltanto la struttura delle cartelle e questa documentazione.

I dati si scaricano dal
[Drive del gruppo](https://drive.google.com/drive/folders/1m_ZYEXbEmvKSD3bNqHsM3GS07wqPDyTT?usp=drive_link)
e vanno scompattati qui mantenendo i percorsi descritti sotto, che sono gli stessi che gli
script si aspettano per default (`src/utils/paths.py`).

## Due corpus, due varianti

Gli esperimenti girano su due corpus, distinti a ogni livello dell'albero.

- **`subset/`** — corpus ridotto, circa 15.000 scene. È quello su cui è stata fatta la
  messa a punto.
- **`fullset/`** — corpus completo, circa 55.000 scene.

Dentro ciascuno, gli scene graph esistono in due varianti che servono a isolare il
contributo del ragionamento ontologico.

- **`semantic/`** — grafi arricchiti dalle inferenze di Virtuoso.
- **`baseline/`** — grafi con le sole annotazioni GQA, senza arricchimento.

## Struttura

```
data/
├── images/
│   ├── subset/
│   │   ├── images_all_15k/     immagini del corpus ridotto
│   │   ├── imagesTest/         sole immagini delle test_queries
│   │   └── imageVectorDB/      gallery indicizzata dalla demo
│   └── fullset/
│       ├── train/  val/  Test/
│       └── VectorDB/           gallery del corpus completo
│
├── sceneGraph/
│   ├── subset/{semantic,baseline}/{raw,embedded}/
│   ├── fullset/{semantic,baseline}/{raw,embedded}/
│   └── json/                   export leggibile dei grafi, per la demo e il VLM
│
└── models/
    ├── subset/
    │   ├── checkpoints/        pesi del graph encoder (.pth)
    │   ├── semantic_web/gcn/   embedding estratti dai modelli semantici
    │   ├── baseline/           embedding dei modelli senza arricchimento
    │   └── faiss/              indici FAISS precalcolati
    └── fullset/
        ├── checkpoints/
        ├── semantic_web/gcn/
        ├── baseline/{gcn,vision}/
        └── faiss/
```

`raw/` contiene i grafi senza feature di nodo e arco; `embedded/` gli stessi grafi con
`x` = embedding NOMIC dei nodi (768 dim) e `edge_attr` = embedding NOMIC delle relazioni.
La pipeline lavora sempre sui file in `embedded/`.

Ogni split è un unico file `.pt` con la lista di oggetti `Data` di PyTorch Geometric:
`{train,val,test_gallery,test_queries}_scene_graphs.pt`, divisi con seed fisso e
condivisi da tutti i modelli, comprese le baseline visive.

## Formato dei file di embedding

Tutti gli embedding, sia della GCN sia delle baseline visive, usano lo stesso dizionario,
in modo che la valutazione sia un'unica pipeline.

```python
{"embeddings": Tensor[N, D], "image_ids": [...], "dim": D, "split": str, "encoder": str}
```

`relevance_test.pt` è invece la ground truth di rilevanza query→gallery: matrice di score
continui (`relevance`), versione a soglia (`binary`) e parametri usati (`meta`). Il formato
è documentato in [`src/evaluation/build_relevance.py`](../src/evaluation/build_relevance.py).

## Come si rigenera

| Cosa | Comando |
|---|---|
| Scene graph da Virtuoso | `python -m src.datasets.dataset_builder` |
| Embedding NOMIC | `python -m src.datasets.add_nomic_embeddings` |
| Split | `python -m src.datasets.make_splits` |
| Checkpoint | `python -m src.training.train --config experiments/configs/<esperimento>.yaml` |
| Embedding GCN | `python -m src.evaluation.extract_gcn_embeddings --config experiments/configs/extract_gine_triplet.yaml` |
| Baseline visive | `python -m src.evaluation.extract_baseline_embeddings --config experiments/configs/vision_baseline_resnet.yaml` |
| Ground truth | `python -m src.evaluation.build_relevance --config experiments/configs/relevance.yaml` |
| Indici FAISS | `python -m src.evaluation.build_faiss_index` |
