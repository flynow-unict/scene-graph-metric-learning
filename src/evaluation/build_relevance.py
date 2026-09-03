"""Ground-truth di rilevanza query->gallery per la valutazione del retrieval.

Utilizza la stessa metrica di similarita' definita per i positivi del training (scene_similarity.py):
per ogni scena-query calcola quanto ogni scena-gallery risulti "semanticamente simile",
usando una Jaccard mista pesata IDF + veto semantico NOMIC. Questo garantisce che la
fase di valutazione utilizzi lo stesso criterio appreso dal modello, evitando disallineamenti.

L'output e' salvato in un dizionario (tramite torch.save) con una struttura generica,
in modo da poter calcolare metriche come Precision/Recall@K, NDCG e MAP senza dover
rilanciare questo script:

    {
      "query_ids":   [id, ...],                 # len Q, ordine delle righe
      "gallery_ids": [id, ...],                 # len G, ordine delle colonne
      "relevance":   FloatTensor[Q, G],         # score continuo in [0,1] di similarita'
      "binary":      BoolTensor[Q, G],          # relevance >= bin_threshold
      "meta": {... parametri usati ...}
    }

- score continuo -> metriche graded (NDCG, correlazione di rank);
- binary          -> metriche set-based (Precision/Recall/MAP@K).
Eventuali altre soglie binarie possono essere ricavate direttamente dal tensore `relevance`.

    python -m src.evaluation.build_relevance --config experiments/configs/relevance.yaml
"""
import argparse
import os

import torch

from src.datasets.graph_dataset import load_dataset
from src.evaluation.scene_similarity import (
    graph_signature, compute_idf, similarity_from_signatures,
    compute_scene_embeddings, build_inverted_index)
from src.utils.config import parse_with_config
from src.utils import paths


def build_relevance(queries, gallery, mode="mixed", alpha=0.5, min_sim=0.15,
                    sem_veto=0.55, sim_strong=0.30, min_triples=0):
    """Calcola la matrice di rilevanza [Q, G] utilizzando scene_similarity.

    IDF e centraggio semantico sono calcolati sull'UNIONE di query e gallery:
    lo spazio di valutazione deve essere condiviso, altrimenti i pesi dei token
    e la correzione di anisotropia risulterebbero incoerenti.
    """
    q_sigs = [graph_signature(g) for g in queries]
    g_sigs = [graph_signature(g) for g in gallery]

    # pesi IDF sul corpus unito (stesso metro per entrambi i lati)
    weights = compute_idf(q_sigs + g_sigs)

    # embedding di scena per il veto: centrati sull'unione, poi ri-splittati
    all_emb = compute_scene_embeddings(list(queries) + list(gallery), weights=weights[0])
    q_emb, g_emb = all_emb[:len(queries)], all_emb[len(queries):]

    # indice invertito sugli oggetti della gallery: due scene senza oggetti in comune
    # hanno Jaccard 0 (anche una tripla condivisa implica oggetti condivisi), quindi
    # basta valutare i candidati che condividono >=1 oggetto -> niente O(Q*G) cieco.
    inv = build_inverted_index(g_sigs)

    Q, G = len(queries), len(gallery)
    rel = torch.zeros(Q, G, dtype=torch.float32)
    for i, (obj_i, _tri_i) in enumerate(q_sigs):
        cand = set()
        for o in obj_i:
            cand.update(inv.get(o, ()))
        for j in cand:
            s = similarity_from_signatures(q_sigs[i], g_sigs[j], mode=mode, alpha=alpha,
                                           weights=weights, min_triples=min_triples)
            if s <= min_sim:
                continue
            # veto: i match lessicalmente deboli servono conferma semantica (coseno NOMIC)
            if s < sim_strong and float(q_emb[i] @ g_emb[j]) < sem_veto:
                continue
            rel[i, j] = s
    return rel, q_emb, g_emb


def main(args):
    queries = load_dataset(args.queries)
    gallery = load_dataset(args.gallery)
    print(f"[data] queries={len(queries)} gallery={len(gallery)}")

    rel, _, _ = build_relevance(
        queries, gallery, mode=args.mode, alpha=args.alpha, min_sim=args.min_sim,
        sem_veto=args.sem_veto, sim_strong=args.sim_strong, min_triples=args.min_triples)

    binary = rel >= args.bin_threshold
    per_query = binary.sum(dim=1)
    print(f"[relevance] coppie rilevanti: {int(binary.sum())} totali | "
          f"media {per_query.float().mean():.1f}/query | "
          f"query con >=1 rilevante: {int((per_query > 0).sum())}/{len(queries)}")

    out = {
        "query_ids": [int(g.image_id) for g in queries],
        "gallery_ids": [int(g.image_id) for g in gallery],
        "relevance": rel,
        "binary": binary,
        "meta": {
            "mode": args.mode, "alpha": args.alpha, "min_sim": args.min_sim,
            "sem_veto": args.sem_veto, "sim_strong": args.sim_strong,
            "min_triples": args.min_triples, "bin_threshold": args.bin_threshold,
            "metric": "scene_similarity.py (Jaccard misto pesato IDF + veto NOMIC)",
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(out, args.out)
    print(f"[done] ground truth [{rel.size(0)}x{rel.size(1)}] -> {args.out}")


def build_parser():
    p = argparse.ArgumentParser(description="Ground truth di rilevanza per il retrieval")
    p.add_argument("--queries",
                   default=str(paths.scene_graphs() / "test_queries_scene_graphs.pt"))
    p.add_argument("--gallery",
                   default=str(paths.scene_graphs() / "test_gallery_scene_graphs.pt"))
    p.add_argument("--out",
                   default=str(paths.MODELS_DIR / "fullset" / "semantic_web" / "relevance_test.pt"))
    p.add_argument("--mode", choices=["objects", "triples", "mixed"], default="mixed")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--min-sim", dest="min_sim", type=float, default=0.15)
    p.add_argument("--sem-veto", dest="sem_veto", type=float, default=0.55)
    p.add_argument("--sim-strong", dest="sim_strong", type=float, default=0.30)
    p.add_argument("--min-triples", dest="min_triples", type=int, default=0)
    p.add_argument("--bin-threshold", dest="bin_threshold", type=float, default=0.15,
                   help="soglia di conversione per la matrice binaria (Precision/Recall)")
    return p


if __name__ == "__main__":
    main(parse_with_config(build_parser()))
