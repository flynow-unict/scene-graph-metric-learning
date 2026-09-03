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

def evaluate_cross():
    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir.parent.parent
    cross_dir = base_dir / "data" / "cross_evaluation"
    
    # Configurazioni da valutare (nome_visuale: (percorso_query, percorso_gallery, percorso_ground_truth))
    configs = {}
    gcn_models = ["gine", "sage", "gine_triplet", "sage_triplet"]
    
    # I file appena estratti
    for m in gcn_models:
        configs[f"FullSet Baseline {m.upper()} -> 15k TestSet"] = (
            cross_dir / f"baseline_test_queries_{m}.pt",
            cross_dir / f"baseline_test_gallery_{m}.pt",
            base_dir / "data" / "models" / "subset" / "baseline" / "relevance_test.pt"
        )
        configs[f"FullSet SemanticWeb {m.upper()} -> 15k TestSet"] = (
            cross_dir / f"semantic_test_queries_{m}.pt",
            cross_dir / f"semantic_test_gallery_{m}.pt",
            base_dir / "data" / "models" / "subset" / "semantic_web" / "relevance_test.pt"
        )
        
    k_values = [1, 5, 10, 20]
    results = {}
    
    print("="*60)
    print(" VALUTAZIONE CROSS-DOMAIN (FULLSET -> 15K) ".center(60))
    print("="*60)
    
    for model_name, (query_path, gallery_path, gt_path) in configs.items():
        if not (query_path.exists() and gallery_path.exists() and gt_path.exists()):
            print(f"[!] Dati mancanti per {model_name}, salto.")
            continue
            
        print(f"\n[*] Analisi modello: {model_name}")
        
        # Carica Ground Truth Originale (15k)
        ground_truth = torch.load(gt_path, weights_only=False)['binary'].numpy()
        
        # Carica embeddings estratti
        query_emb = load_embeddings(query_path)
        gallery_emb = load_embeddings(gallery_path)
        
        dim = gallery_emb.shape[1]
        
        # Normalizzazione
        faiss.normalize_L2(gallery_emb)
        faiss.normalize_L2(query_emb)
        
        # Costruzione e ricerca indice FAISS
        index = faiss.IndexFlatIP(dim)
        index.add(gallery_emb)
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
                if total_relevant > 0: recalls.append(hits_in_k / total_relevant)
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
        
    results_dir = base_dir / "results"
    with open(results_dir / "cross_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    with open(results_dir / "cross_evaluation_results.md", "w") as f:
        f.write("# Risultati Cross-Evaluation: Full Set Models -> 15k Test Set\n\n")
        f.write("Questo report valuta la generalizzazione (Zero-Shot) dei modelli addestrati sull'intero dataset (55k) applicati alle immagini del sotto-dataset (15k).\n\n")
        for model, metrics in results.items():
            f.write(f"## Modello: `{model}`\n")
            f.write("| K | Precision (%) | Recall (%) | Accuracy/HitRate (%) |\n")
            f.write("|---|---|---|---|\n")
            for k_str, vals in metrics.items():
                k_val = k_str.split("=")[1]
                f.write(f"| {k_val} | {vals['Precision']} | {vals['Recall']} | {vals['Accuracy_HitRate']} |\n")
            f.write("\n")
            
    print(f"\n[✔] Valutazione completata. Risultati salvati in results/cross_evaluation_results.md")

if __name__ == "__main__":
    evaluate_cross()
