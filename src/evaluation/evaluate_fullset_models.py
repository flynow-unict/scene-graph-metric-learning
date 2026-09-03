import os
import json
import torch
import faiss
import numpy as np
from pathlib import Path

def load_embeddings(filepath):
    data = torch.load(filepath, weights_only=False)
    if 'embeddings' in data:
        return data['embeddings'].numpy()
    return data.numpy()

def evaluate_fullset():
    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir.parent.parent
    fullset_dir = base_dir / "data" / "models" / "fullset"
    faiss_dir = fullset_dir / "faiss"
    
    # Ground truths
    rel_baseline = fullset_dir / "baseline" / "relevance_test.pt"
    rel_semantic = fullset_dir / "semantic_web" / "relevance_test.pt"
    
    k_values = [1, 5, 10, 20]
    results = {}
    
    # Mappa dei modelli da valutare (nome_visuale: (percorso_query, percorso_indice_faiss, path_relevance))
    configs = {
        # 1. Baseline GCN (NoRel)
        "Baseline GCN (GINE)": (
            fullset_dir / "baseline" / "gcn" / "baseline_gcn_test_queries_gine.pt",
            faiss_dir / "baseline_gcn_gine.index",
            rel_baseline
        ),
        "Baseline GCN (SAGE)": (
            fullset_dir / "baseline" / "gcn" / "baseline_gcn_test_queries_sage.pt",
            faiss_dir / "baseline_gcn_sage.index",
            rel_baseline
        ),
        "Baseline GCN (GINE TRIPLET)": (
            fullset_dir / "baseline" / "gcn" / "baseline_gcn_test_queries_gine_triplet.pt",
            faiss_dir / "baseline_gcn_gine_triplet.index",
            rel_baseline
        ),
        "Baseline GCN (SAGE TRIPLET)": (
            fullset_dir / "baseline" / "gcn" / "baseline_gcn_test_queries_sage_triplet.pt",
            faiss_dir / "baseline_gcn_sage_triplet.index",
            rel_baseline
        ),
        # 2. Semantic Web GCN
        "Semantic Web GCN (GINE)": (
            fullset_dir / "semantic_web" / "gcn" / "gcn_test_queries_gine.pt",
            faiss_dir / "semantic_web_gcn_gine.index",
            rel_semantic
        ),
        "Semantic Web GCN (SAGE)": (
            fullset_dir / "semantic_web" / "gcn" / "gcn_test_queries_sage.pt",
            faiss_dir / "semantic_web_gcn_sage.index",
            rel_semantic
        ),
        "Semantic Web GCN (GINE TRIPLET)": (
            fullset_dir / "semantic_web" / "gcn" / "gcn_test_queries_gine_triplet.pt",
            faiss_dir / "semantic_web_gcn_gine_triplet.index",
            rel_semantic
        ),
        "Semantic Web GCN (SAGE TRIPLET)": (
            fullset_dir / "semantic_web" / "gcn" / "gcn_test_queries_sage_triplet.pt",
            faiss_dir / "semantic_web_gcn_sage_triplet.index",
            rel_semantic
        ),
        # 3. Vision Baselines
        "Vision Baseline (CLIP)": (
            fullset_dir / "baseline" / "vision" / "clip_test_queries.pt",
            faiss_dir / "vision_clip.index",
            rel_baseline
        ),
        "Vision Baseline (RESNET)": (
            fullset_dir / "baseline" / "vision" / "resnet_test_queries.pt",
            faiss_dir / "vision_resnet.index",
            rel_baseline
        )
    }
    
    print("="*60)
    print(" VALUTAZIONE RETRIEVAL - FULL SET (55k) ".center(60))
    print("="*60)
    
    for model_name, (query_path, index_path, relevance_path) in configs.items():
        if not (query_path.exists() and index_path.exists()):
            print(f"[!] Dati mancanti per {model_name}, salto la valutazione.")
            continue
            
        print(f"\n[*] Analisi modello: {model_name}")
        
        try:
            # Carica il database FAISS precalcolato
            index = faiss.read_index(str(index_path))
            
            # Carica la ground truth
            ground_truth = torch.load(relevance_path, weights_only=False)['binary'].numpy()
            
            # Carica e normalizza le query
            query_emb = load_embeddings(query_path)
            faiss.normalize_L2(query_emb)

            
            max_k = max(k_values)
            D, I = index.search(query_emb, max_k)
            
            model_results = {}
            
            for k in k_values:
                precisions = []
                recalls = []
                hits = []
                
                for q_idx in range(query_emb.shape[0]):
                    retrieved_indices = I[q_idx, :k]
                    rel_scores = ground_truth[q_idx, retrieved_indices]
                    total_relevant = ground_truth[q_idx, :].sum()
                    
                    hits_in_k = rel_scores.sum()
                    precisions.append(hits_in_k / k)
                    
                    if total_relevant > 0:
                        recalls.append(hits_in_k / total_relevant)
                        
                    hits.append(1 if hits_in_k > 0 else 0)
                    
                avg_precision = np.mean(precisions) * 100
                avg_recall = np.mean(recalls) * 100
                hit_rate = np.mean(hits) * 100
                
                model_results[f"K={k}"] = {
                    "Precision": round(avg_precision, 2),
                    "Recall": round(avg_recall, 2),
                    "Accuracy_HitRate": round(hit_rate, 2)
                }
                
                print(f"    @K={k:<2} | Precision: {avg_precision:5.2f}% | Recall: {avg_recall:5.2f}% | Accuracy: {hit_rate:5.2f}%")
                
            results[model_name] = model_results
        except Exception as e:
            print(f"[!] Errore durante la valutazione di {model_name}: {e}")
            
    # Salvataggio dei risultati in un file JSON
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_json = results_dir / "fullset_baseline_evaluation_results.json"
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)
        
    # Creazione di un file Markdown leggibile
    output_md = results_dir / "fullset_baseline_evaluation_results.md"
    with open(output_md, "w") as f:
        f.write("# Risultati Retrieval - Full Set Baseline (55k)\n\n")
        f.write("Questo report mostra le metriche GCN della nuova Baseline Pura sul Full Set.\n\n")
        for model, metrics in results.items():
            f.write(f"## Modello: `{model}`\n")
            f.write("| K | Precision (%) | Recall (%) | Accuracy/HitRate (%) |\n")
            f.write("|---|---|---|---|\n")
            for k_str, vals in metrics.items():
                k_val = k_str.split("=")[1]
                f.write(f"| {k_val} | {vals['Precision']} | {vals['Recall']} | {vals['Accuracy_HitRate']} |\n")
            f.write("\n")
            
    print(f"\n[✔] Valutazione completata. Risultati salvati in:")
    print(f"    - {output_json.name}")
    print(f"    - {output_md.name}")

if __name__ == "__main__":
    evaluate_fullset()
