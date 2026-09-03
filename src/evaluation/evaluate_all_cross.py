import torch
import faiss
from pathlib import Path

def load_embeddings(filepath):
    data = torch.load(filepath, map_location='cpu', weights_only=False)
    if 'node_embeddings' in data:
        embeddings = data['node_embeddings'].numpy()
    elif 'embeddings' in data:
        embeddings = data['embeddings'].numpy()
    else:
        raise KeyError(f"Nessuna chiave valida trovata in {filepath}")
    return embeddings

def evaluate(query_emb, index, ground_truth, k_values):
    max_k = max(k_values)
    D, I = index.search(query_emb, max_k)
    
    num_queries = query_emb.shape[0]
    metrics = {f"K={k}": {"Precision": 0.0, "Recall": 0.0, "Accuracy_HitRate": 0.0} for k in k_values}
    
    for k in k_values:
        total_precision = 0.0
        total_recall = 0.0
        hits = 0
        
        for q_idx in range(num_queries):
            retrieved_indices = I[q_idx, :k]
            relevant_items = set(torch.where(torch.tensor(ground_truth[q_idx]) > 0)[0].tolist())
            retrieved_set = set(retrieved_indices.tolist())
            
            true_positives = len(relevant_items.intersection(retrieved_set))
            
            if len(relevant_items) > 0:
                precision = true_positives / k
                recall = true_positives / len(relevant_items)
                total_precision += precision
                total_recall += recall
                if true_positives > 0:
                    hits += 1
                    
        avg_precision = (total_precision / num_queries) * 100
        avg_recall = (total_recall / num_queries) * 100
        hit_rate = (hits / num_queries) * 100
        
        metrics[f"K={k}"]["Precision"] = round(avg_precision, 2)
        metrics[f"K={k}"]["Recall"] = round(avg_recall, 2)
        metrics[f"K={k}"]["Accuracy_HitRate"] = round(hit_rate, 2)
        
    return metrics

def run_evaluations(title, configs, out_md):
    print("="*60)
    print(title.center(60))
    print("="*60)
    
    md_content = f"# {title}\n\n"
    
    for model_name, (query_path, gallery_path, relevance_path) in configs.items():
        if not query_path.exists() or not gallery_path.exists():
            print(f"[!] Dati mancanti per {model_name}, salto.")
            continue
            
        if not relevance_path.exists():
            print(f"[!] ERRORE: Ground truth mancante: {relevance_path}")
            continue
            
        print(f"\n[*] Analisi: {model_name}")
        
        try:
            ground_truth = torch.load(relevance_path, weights_only=False)['binary'].numpy()
            
            query_emb = load_embeddings(query_path)
            gallery_emb = load_embeddings(gallery_path)
            
            faiss.normalize_L2(query_emb)
            faiss.normalize_L2(gallery_emb)
            
            dim = gallery_emb.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(gallery_emb)
            
            metrics = evaluate(query_emb, index, ground_truth, [1, 5, 10, 20])
            
            md_content += f"## Modello: `{model_name}`\n"
            md_content += "| K | Precision (%) | Recall (%) | Accuracy/HitRate (%) |\n"
            md_content += "|---|---|---|---|\n"
            
            for k in [1, 5, 10, 20]:
                k_str = f"K={k}"
                p = metrics[k_str]['Precision']
                r = metrics[k_str]['Recall']
                a = metrics[k_str]['Accuracy_HitRate']
                print(f"    @K={k:<2} | Precision: {p:>5.2f}% | Recall: {r:>5.2f}% | Accuracy: {a:>5.2f}%")
                md_content += f"| {k} | {p} | {r} | {a} |\n"
                
            md_content += "\n"
        except Exception as e:
            print(f"[!] Errore su {model_name}: {e}")
            
    with open(out_md, 'w') as f:
        f.write(md_content)
    print(f"\n[✔] Risultati salvati in: {out_md}")


if __name__ == "__main__":
    out_cross_dir = Path("data/models/cross_evaluation")
    out_rev_dir = Path("data/models/reverse_cross_evaluation")
    
    gt_subset_base = Path("data/models/subset/baseline/relevance_test.pt")
    gt_subset_sem = Path("data/models/subset/semantic_web/relevance_test.pt")
    gt_full_base = Path("data/models/fullset/baseline/relevance_test.pt")
    gt_full_sem = Path("data/models/fullset/semantic_web/relevance_test.pt")
    
    # 1. CROSS EVALUATION (Modelli Fullset su Dati Subset -> Ground Truth = Subset)
    cross_configs = {}
    for arch in ["gine", "gine_triplet", "sage", "sage_triplet"]:
        cross_configs[f"Fullset Semantic Web GCN ({arch.upper()})"] = (
            out_cross_dir / f"semantic_test_queries_{arch}.pt",
            out_cross_dir / f"semantic_test_gallery_{arch}.pt",
            gt_subset_sem
        )
        cross_configs[f"Fullset Baseline GCN ({arch.upper()})"] = (
            out_cross_dir / f"baseline_test_queries_{arch}.pt",
            out_cross_dir / f"baseline_test_gallery_{arch}.pt",
            gt_subset_base
        )
        
    run_evaluations(
        "Risultati Cross-Evaluation (Modelli Full Set su Dati Subset 15k)", 
        cross_configs, 
        "results/cross_evaluation_new_results.md"
    )
    
    # 2. REVERSE CROSS EVALUATION (Modelli Subset su Dati Fullset -> Ground Truth = Fullset)
    rev_configs = {}
    for arch in ["gine", "gine_triplet", "sage", "sage_triplet"]:
        rev_configs[f"Subset Semantic Web GCN ({arch.upper()})"] = (
            out_rev_dir / f"semantic_test_queries_{arch}.pt",
            out_rev_dir / f"semantic_test_gallery_{arch}.pt",
            gt_full_sem
        )
        rev_configs[f"Subset Baseline GCN ({arch.upper()})"] = (
            out_rev_dir / f"baseline_test_queries_{arch}.pt",
            out_rev_dir / f"baseline_test_gallery_{arch}.pt",
            gt_full_base
        )
        
    run_evaluations(
        "Risultati Reverse Cross-Evaluation (Modelli Subset su Dati Full Set 55k)", 
        rev_configs, 
        "results/reverse_cross_evaluation_new_results.md"
    )
