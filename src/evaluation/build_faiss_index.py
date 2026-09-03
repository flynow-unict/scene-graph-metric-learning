import os
import torch
import faiss
from pathlib import Path

def load_embeddings(filepath):
    data = torch.load(filepath)
    return data['embeddings'].numpy()

def build_indices():
    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir.parent.parent
    gcn_dir = base_dir / "data" / "models" / "gcn"
    
    # Modelli di Dario disponibili
    models = ["gcn_gine", "gcn_sage", "gcn_gine_triplet", "gcn_sage_triplet"]
    
    print("="*50)
    print(" CREAZIONE DATABASE FAISS ".center(50))
    print("="*50)
    
    faiss_dir = base_dir / "data" / "models" / "faiss"
    faiss_dir.mkdir(parents=True, exist_ok=True)
    
    for model in models:
        gallery_path = gcn_dir / f"{model}_test_gallery.pt"
        index_path = faiss_dir / f"faiss_{model}.index"
        
        if not gallery_path.exists():
            print(f"[!] File saltato: {gallery_path.name} non trovato.")
            continue
            
        print(f"[*] Elaborazione {model}...")
        gallery_emb = load_embeddings(gallery_path)
        dim = gallery_emb.shape[1]
        
        # Normalizzazione L2 per far sì che FAISS Inner Product calcoli la Cosine Similarity
        faiss.normalize_L2(gallery_emb)
        
        # Creiamo l'indice in RAM
        index = faiss.IndexFlatIP(dim)
        index.add(gallery_emb)
        
        # Salviamo l'indice su disco per l'API di Santi e per le query successive
        faiss.write_index(index, str(index_path))
        print(f"    -> Indice salvato in: {index_path.name} ({index.ntotal} vettori)")

if __name__ == "__main__":
    build_indices()
    print("\n[✔] Creazione di tutti gli indici FAISS completata.")
