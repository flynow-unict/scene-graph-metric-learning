import os
import torch
import faiss
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.data import Batch, Data
from torch_geometric.utils import subgraph

# Assicuriamoci che python trovi il modulo script.DeepLearning
from src.evaluation.extract_gcn_embeddings import build_model_from_ckpt
from src.utils.device import get_device

def drop_nodes_randomly(data, p=0.2):
    """Rimuove strutturalmente il p% dei nodi dal grafo."""
    if p == 0.0:
        return data
        
    num_nodes = data.num_nodes
    if num_nodes == 0:
        return data
        
    # Maschera booleana dei nodi da TENERE
    keep_mask = torch.rand(num_nodes) > p
    
    # Preveniamo la distruzione totale (teniamo almeno 1 nodo)
    if not keep_mask.any():
        keep_mask[torch.randint(0, num_nodes, (1,))] = True
        
    keep_indices = keep_mask.nonzero(as_tuple=False).view(-1)
    
    # 1. Filtriamo i Node Features
    new_x = data.x[keep_mask]
    
    # 2. Filtriamo gli Archi (manteniamo solo gli archi tra nodi superstiti)
    if data.edge_index is not None and data.edge_index.numel() > 0:
        new_edge_index, new_edge_attr = subgraph(
            keep_indices, 
            data.edge_index, 
            edge_attr=getattr(data, 'edge_attr', None), 
            relabel_nodes=True,
            num_nodes=num_nodes
        )
    else:
        new_edge_index = torch.empty((2, 0), dtype=torch.long)
        new_edge_attr = torch.empty((0, data.edge_attr.size(1))) if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        
    return Data(x=new_x, edge_index=new_edge_index, edge_attr=new_edge_attr, image_id=data.image_id)

@torch.no_grad()
def extract_embeddings(model, graphs, device, batch_size=128):
    """Estrae i vettori passandoli nella GCN"""
    model.eval()
    all_emb = []
    
    for i in range(0, len(graphs), batch_size):
        chunk = graphs[i:i + batch_size]
        batch = Batch.from_data_list(chunk).to(device)
        
        edge_attr = getattr(batch, "edge_attr", None)
        z = model(batch.x, batch.edge_index, batch.batch, edge_attr).cpu()
        all_emb.append(z)
        
    embeddings = torch.cat(all_emb, dim=0)
    faiss.normalize_L2(embeddings.numpy())
    return embeddings.numpy()

def main():
    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir.parent.parent
    
    model_name = "gcn_gine_triplet"
    ckpt_path = base_dir / "data" / "models" / "fullset" / "checkpoints" / "gcn_encoder_gine_triplet.pth"
    graphs_path = base_dir / "test_francesco" / "robustness_sample.pt"
    index_path = base_dir / "data" / "models" / "fullset" / "faiss" / "semantic_web_gcn_gine_triplet.index"
    relevance_path = base_dir / "data" / "models" / "fullset" / "semantic_web" / "relevance_test.pt"
    
    print("="*60)
    print(" AVVIO ROBUSTNESS TEST (SELF-RETRIEVAL) ".center(60))
    print("="*60)
    
    device = get_device()
    print(f"[*] Caricamento pesi GCN da: {ckpt_path.name} su {device}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model_from_ckpt(ckpt, device)
    
    print(f"[*] Caricamento sample grafi...")
    original_graphs = torch.load(graphs_path, weights_only=False)
    
    print(f"[*] Caricamento FAISS Index e ID della Gallery...")
    index = faiss.read_index(str(index_path))
    
    # Carichiamo la relevance_test solo per estrarre la gallery_ids
    gt_data = torch.load(relevance_path, weights_only=False)
    gallery_ids = gt_data.get('gallery_ids', [])
    if isinstance(gallery_ids, torch.Tensor):
        gallery_ids = gallery_ids.tolist()
    elif isinstance(gallery_ids, np.ndarray):
        gallery_ids = gallery_ids.tolist()
    
    # Mappiamo FAISS index row -> image_id stringa
    idx_to_gallery_id = {i: str(g_id) for i, g_id in enumerate(gallery_ids)}
    
    drop_rates = [0.0, 0.1, 0.2, 0.3, 0.4]
    accuracies = [] # K=20
    precisions = [] # K=1
    
    for p in drop_rates:
        print(f"\n--- Test Dropout Nodi: {int(p*100)}% ---")
        
        # Perturbiamo i grafi
        corrupted_graphs = [drop_nodes_randomly(g, p=p) for g in original_graphs]
        query_emb = extract_embeddings(model, corrupted_graphs, device)
        
        K_max = 20
        D, I = index.search(query_emb, K_max)
        
        hits_k20 = []
        hits_k1 = []
        
        for q_idx in range(query_emb.shape[0]):
            target_id = str(original_graphs[q_idx].image_id)
            
            retrieved_k20 = [idx_to_gallery_id.get(idx, "") for idx in I[q_idx, :20]]
            retrieved_k1 = [idx_to_gallery_id.get(idx, "") for idx in I[q_idx, :1]]
            
            hits_k20.append(1 if target_id in retrieved_k20 else 0)
            hits_k1.append(1 if target_id in retrieved_k1 else 0)
            
        acc = np.mean(hits_k20) * 100
        prec = np.mean(hits_k1) * 100
        
        print(f"    Accuracy @20 (Self-Retrieval):  {acc:.2f}%")
        print(f"    Precision @1 (Self-Retrieval): {prec:.2f}%")
        
        accuracies.append(acc)
        precisions.append(prec)
        
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    plt.plot([p*100 for p in drop_rates], accuracies, marker='o', label='Accuracy @ K=20', linewidth=2)
    plt.plot([p*100 for p in drop_rates], precisions, marker='s', label='Precision @ K=1', linewidth=2)
    
    plt.title('Robustness Test: Degrado Self-Retrieval vs Nodi Rimosi', fontsize=14)
    plt.xlabel('Percentuale di Nodi Rimosi (%)', fontsize=12)
    plt.ylabel('Score Metrica (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    
    plot_path = results_dir / "robustness_test_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n[✔] Test completato. Grafico salvato in: {plot_path.name}")

if __name__ == "__main__":
    main()

