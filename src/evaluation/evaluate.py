import json
import torch
import faiss
import numpy as np
from pathlib import Path

def load_embeddings(filepath):
    data = torch.load(filepath)
    return data['embeddings'].numpy()

def evaluate_models():
    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir.parent.parent
    gcn_dir = base_dir / "data" / "models" / "gcn"
    
    models = ["gcn_gine", "gcn_sage", "gcn_gine_triplet", "gcn_sage_triplet"]
    k_values = [1, 5, 10, 20]
    
    relevance_path = gcn_dir / "relevance_test.pt"
    if not relevance_path.exists():
        print(f"[!] ERRORE: Ground truth non trovata in {relevance_path}")
        return
        
    ground_truth = torch.load(relevance_path)['binary'].numpy()
    
    results = {}
    
    print("="*60)
    print(" VALUTAZIONE METRICHE SU TUTTI I MODELLI FAISS ".center(60))
    print("="*60)
    
    faiss_dir = base_dir / "data" / "models" / "faiss"
    
    for model in models:
        index_path = faiss_dir / f"faiss_{model}.index"
        query_path = gcn_dir / f"{model}_test_queries.pt"
        
        if not (index_path.exists() and query_path.exists()):
            print(f"[!] Dati mancanti per {model}, salto la valutazione.")
            continue
            
        print(f"\n[*] Analisi modello: {model.upper()}")
        
        # Carica il database FAISS precalcolato
        index = faiss.read_index(str(index_path))
        
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
            
        results[model] = model_results
        
    # Salvataggio dei risultati in un file JSON
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_json = results_dir / "evaluation_results.json"
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)
        
    # Creazione di un file Markdown leggibile
    output_md = results_dir / "evaluation_results.md"
    with open(output_md, "w") as f:
        f.write("# Risultati Retrieval Modelli GCN\n\n")
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
    evaluate_models()
