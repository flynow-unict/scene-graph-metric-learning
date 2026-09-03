import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from torch_geometric.utils import to_networkx

# Assicuriamoci che python trovi il modulo script.DeepLearning
from src.evaluation.extract_gcn_embeddings import build_model_from_ckpt
from src.utils.device import get_device

# Non usiamo più il wrapper: PyG supporta la regressione vettoriale diretta.

def explain_graph(graph_idx=42):
    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir.parent.parent
    
    ckpt_path = base_dir / "data" / "models" / "fullset" / "checkpoints" / "gcn_encoder_gine_triplet.pth"
    graphs_path = base_dir / "test_francesco" / "explainability_samples.pt"
    
    device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model_from_ckpt(ckpt, device)
    model.eval()
    
    print(f"[*] Caricamento query grafi...")
    graphs = torch.load(graphs_path, weights_only=False)
    
    # Troviamo un grafo interessante (con un po' di nodi e relazioni)
    graph = None
    for g in graphs:
        if g.num_nodes > 5 and g.edge_index.size(1) > 5:
            graph = g.to(device)
            break
            
    if graph is None:
        graph = graphs[0].to(device)
        
    print(f"[*] Grafo selezionato: ID {graph.image_id} con {graph.num_nodes} nodi e {graph.edge_index.size(1)} archi.")
    
    # 1. Calcoliamo l'embedding originale (Target)
    with torch.no_grad():
        edge_attr = getattr(graph, 'edge_attr', None)
        # Assicuriamoci che il modello usi un batch fittizio
        batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
        target_emb = model(graph.x, graph.edge_index, batch, edge_attr)
        
    # 2. Inizializziamo GNNExplainer per regressione sul vettore 256-D
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=300, lr=0.01),
        explanation_type='model',
        node_mask_type='object',
        edge_mask_type='object',
        model_config=ModelConfig(
            mode='regression',
            task_level='graph',
            return_type='raw',
        ),
    )
    
    print(f"[*] Esecuzione GNNExplainer per trovare i nodi chiave...")
    # Passiamo esplicitamente batch e target
    explanation = explainer(
        x=graph.x, 
        edge_index=graph.edge_index, 
        batch=batch, 
        edge_attr=edge_attr,
        target=target_emb
    )
    
    # 3. Estraiamo i risultati (usiamo squeeze per avere un array 1D piatto)
    node_masks = explanation.node_mask.cpu().numpy().squeeze()
    
    print("\n" + "="*50)
    print(" REPORT EXPLAINABILITY ".center(50))
    print("="*50)
    
    # Stampiamo i top 5 nodi più importanti
    important_indices = np.argsort(node_masks)[::-1]
    
    print("Top 5 oggetti che definiscono questa scena per il modello:")
    for i in range(min(5, len(important_indices))):
        idx = int(important_indices[i].item()) if hasattr(important_indices[i], 'item') else int(important_indices[i])
        score = float(node_masks[idx].item()) if hasattr(node_masks[idx], 'item') else float(node_masks[idx])
        text = graph.node_text[idx]
        # Estraiamo solo il nome base prima di 'Categories:'
        obj_name = text.split('.')[0] if '.' in text else text
        print(f"  {i+1}. [{score:.3f}] {obj_name}")
        
    # 4. Creiamo una visualizzazione
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    G = to_networkx(graph, to_undirected=True)
    plt.figure(figsize=(10, 8))
    
    # Colori basati sull'importanza
    node_color = [node_masks[i] for i in range(graph.num_nodes)]
    labels = {i: graph.node_text[i].split('.')[0] for i in range(graph.num_nodes)}
    
    pos = nx.spring_layout(G, k=1.0) # Layout del grafo
    vmin, vmax = float(node_masks.min()), float(node_masks.max())
    nodes = nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_color, cmap=plt.cm.Reds, vmin=vmin, vmax=vmax)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight='bold')
    
    plt.colorbar(nodes, label='Importanza GNNExplainer')
    plt.title(f"GNNExplainer - Scena {graph.image_id}", fontsize=14)
    plt.axis('off')
    
    out_path = results_dir / "explainability_graph.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n[✔] Grafo dell'explainability salvato in: {out_path.name}")

if __name__ == "__main__":
    explain_graph()
