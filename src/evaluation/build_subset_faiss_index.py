import os
import torch
import faiss
from pathlib import Path

def load_embeddings(filepath):
    data = torch.load(filepath, weights_only=False)
    if 'embeddings' in data:
        return data['embeddings'].numpy()
    return data.numpy()

def build_indices():
    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir.parent.parent
    subset_dir = base_dir / "data" / "models" / "subset"
    
    # Cartella di output
    faiss_dir = subset_dir / "faiss"
    faiss_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print(" CREAZIONE DATABASE FAISS - SUBSET (15k) ".center(60))
    print("="*60)
    
    gcn_models = ["gine", "sage", "gine_triplet", "sage_triplet"]
    
    # 1. Semantic Web GCN (Rel)
    # Si salvano come faiss_gcn_{m}.index (compatibile con workerAI)
    for m in gcn_models:
        gallery_path = subset_dir / "semantic_web" / "gcn" / f"gcn_{m}_test_gallery.pt"
        index_path = faiss_dir / f"faiss_gcn_{m}.index"
        process_index(gallery_path, index_path, f"Semantic Web Subset GCN ({m.upper()})")

    # 2. Baseline GCN (NoRel)
    # Si salvano come baseline_gcn_{m}.index (compatibile con workerAI)
    for m in gcn_models:
        gallery_path = subset_dir / "baseline" / "gcn" / f"baseline_gcn_test_gallery_{m}.pt"
        index_path = faiss_dir / f"baseline_gcn_{m}.index"
        process_index(gallery_path, index_path, f"Baseline Subset GCN ({m.upper()})")
        
    # 3. Vision Baselines
    # Si salvano come vision_{m}.index (compatibile con workerAI)
    for m in ["clip", "resnet"]:
        gallery_path = subset_dir / "baseline" / f"{m}_test_gallery.pt"
        index_path = faiss_dir / f"vision_{m}.index"
        process_index(gallery_path, index_path, f"Vision Baseline ({m.upper()})")

def process_index(gallery_path, index_path, model_name):
    if not gallery_path.exists():
        print(f"[!] Saltato: {model_name} (File non trovato: {gallery_path.name})")
        return
        
    print(f"[*] Elaborazione {model_name}...")
    try:
        gallery_emb = load_embeddings(gallery_path)
        dim = gallery_emb.shape[1]
        
        # Normalizzazione per Cosine Similarity
        faiss.normalize_L2(gallery_emb)
        
        # Creazione IndexFlatIP
        index = faiss.IndexFlatIP(dim)
        index.add(gallery_emb)
        
        # Salvataggio su disco
        faiss.write_index(index, str(index_path))
        print(f"    -> Salvato: {index_path.name} ({index.ntotal} vettori indicizzati in RAM)")
    except Exception as e:
        print(f"[!] Errore durante l'elaborazione di {model_name}: {e}")

if __name__ == "__main__":
    build_indices()
    print("\n[✔] Creazione di tutti gli indici FAISS completata.")
