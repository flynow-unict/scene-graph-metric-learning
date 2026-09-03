import torch
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch_geometric

# Configurazione percorsi
script_dir = os.path.dirname(os.path.abspath(__file__))
script_folder = os.path.dirname(script_dir)
project_dir = os.path.dirname(script_folder)

# Directory input e output
input_dir = os.path.join(project_dir, "data", "sceneGraph", "raw", "full_set")
output_dir = os.path.join(project_dir, "data", "sceneGraph", "embedded", "full_set")
os.makedirs(output_dir, exist_ok=True)

# I quattro split generati precedentemente
splits = [
    "train_scene_graphs.pt",
    "val_scene_graphs.pt",
    "test_gallery_scene_graphs.pt",
    "test_queries_scene_graphs.pt"
]

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

def process_split(model, split_filename, device):
    input_file = os.path.join(input_dir, split_filename)
    output_file = os.path.join(output_dir, split_filename)
    
    if not os.path.exists(input_file):
        print(f"[!] File {input_file} non trovato, split saltato.")
        return
        
    print(f"\n--- Elaborazione {split_filename} ---")
    dataset = torch.load(input_file, weights_only=False)
    print(f"[*] Dataset caricato: {len(dataset)} grafi trovati.")
    
    processed_dataset = []
    
    for graph in tqdm(dataset, desc=f"Vettorizzazione {split_filename}"):
        # 1. Nodi: classBase + macroCategorie + attributi
        if hasattr(graph, 'node_text') and graph.node_text:
            # Assicuriamo un prefisso per indicare al modello Nomic che stiamo vettorizzando documenti
            # "search_document: " è il prefisso raccomandato da Nomic per il clustering/stoccaggio
            nomic_docs = [f"search_document: {text}" for text in graph.node_text]
            graph.x = model.encode(nomic_docs, convert_to_tensor=True).cpu()
        else:
            graph.x = torch.empty((0, 768), dtype=torch.float)
            
        # 2. Archi: relazioni raw e dedotte (logica OWL)
        if hasattr(graph, 'edge_text') and graph.edge_text:
            nomic_edges = [f"search_document: {text}" for text in graph.edge_text]
            graph.edge_attr = model.encode(nomic_edges, convert_to_tensor=True).cpu()
        else:
            graph.edge_attr = torch.empty((0, 768), dtype=torch.float)
            
        processed_dataset.append(graph)
        
    print(f"[*] Salvataggio split arricchito in {output_file}...")
    torch.save(processed_dataset, output_file)
    print(f"    -> {split_filename} completato.")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Hardware rilevato: {device.upper()}")
    print(f"[*] Caricamento del modello {MODEL_NAME}...")
    
    # trust_remote_code=True richiesto da nomic-embed-text
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=device)
    
    for split in splits:
        process_split(model, split, device)
        
    print("\n[✔] Vettorizzazione completata su tutti gli split.")

if __name__ == "__main__":
    main()
