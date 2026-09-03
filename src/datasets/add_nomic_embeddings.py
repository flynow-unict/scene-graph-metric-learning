import torch
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch_geometric

# Definiamo i percorsi dei file rispetto allo script attuale
script_dir = os.path.dirname(os.path.abspath(__file__)) # src/datasets
script_folder = os.path.dirname(script_dir)             # Progetto/script
project_dir = os.path.dirname(script_folder)            # Progetto
import argparse

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

def main():
    parser = argparse.ArgumentParser(description="Calcolo Nomic Embeddings per Scene Graph")
    parser.add_argument("--input", type=str, default="", help="File .pt di input")
    parser.add_argument("--output", type=str, default="", help="File .pt di output")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_folder = os.path.dirname(script_dir)
    project_dir = os.path.dirname(script_folder)

    input_file = args.input if args.input else os.path.join(project_dir, "data/sceneGraph/Rel/", "dataset_gqa_scene_graphs.pt")
    output_file = args.output if args.output else os.path.join(project_dir, "data/embedding/Rel/", "dataset_gqa_embedded.pt")

    print(f"[*] Caricamento dataset/grafo da {input_file}...")
    dataset = torch.load(input_file, weights_only=False)
    
    is_single_graph = not isinstance(dataset, list)
    graphs_list = [dataset] if is_single_graph else dataset
    print(f"[*] Grafi trovati: {len(graphs_list)}.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Caricamento del modello {MODEL_NAME} su {device}...")
    
    try:
        model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=device)
    except Exception as e:
        print(f"[!] Avviso rete Nomic: {e}. Caricamento in local_files_only=True...")
        model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=device, local_files_only=True)
    
    print("[*] Inizio estrazione embeddings...")
    processed_dataset = []
    
    for graph in tqdm(graphs_list, desc="Vettorizzazione Grafi"):
        # Vettorizzazione Nodi
        if hasattr(graph, 'node_text') and graph.node_text:
            graph.x = model.encode(graph.node_text, convert_to_tensor=True).cpu()
        else:
            graph.x = torch.empty((0, 768), dtype=torch.float)
            
        # Vettorizzazione Archi
        if hasattr(graph, 'edge_text') and graph.edge_text:
            graph.edge_attr = model.encode(graph.edge_text, convert_to_tensor=True).cpu()
        else:
            graph.edge_attr = torch.empty((0, 768), dtype=torch.float)
            
        processed_dataset.append(graph)
        
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_output = processed_dataset[0] if is_single_graph else processed_dataset
    print(f"[*] Salvataggio grafo arricchito in {output_file}...")
    torch.save(final_output, output_file)
    print("\n[✔] Embeddings Nomic generati e salvati.")

if __name__ == "__main__":
    main()
