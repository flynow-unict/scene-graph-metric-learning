import torch
import os
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch_geometric

# Definiamo i percorsi dei file rispetto allo script attuale
script_dir = os.path.dirname(os.path.abspath(__file__)) # Progetto/script/dataLoader
script_folder = os.path.dirname(script_dir)             # Progetto/script
project_dir = os.path.dirname(script_folder)            # Progetto
input_file = os.path.join(project_dir, "data/sceneGraph/Rel/", "dataset_gqa_scene_graphs.pt")
output_file = os.path.join(project_dir, "data/embedding/Rel/", "dataset_gqa_embedded.pt")

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pt", type=str, default="", help="Input graph (.pt) for single inference")
    parser.add_argument("--output_pt", type=str, default="", help="Output embedded graph (.pt) for single inference")
    args = parser.parse_args()

    # Determina i file di input/output
    current_input = args.input_pt if args.input_pt else input_file
    current_output = args.output_pt if args.output_pt else output_file

    print(f"Caricamento dati da {current_input}...")
    dataset_or_graph = torch.load(current_input, weights_only=False)
    
    # Normalizza in una lista per un loop uniforme
    is_single_graph = not isinstance(dataset_or_graph, list)
    dataset = [dataset_or_graph] if is_single_graph else dataset_or_graph
    
    print(f"Dati caricati: {len(dataset)} grafi trovati.")
    
    # Rilevamento automatico della GPU per SLURM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Caricamento del modello {MODEL_NAME} su {device}...")
    
    # trust_remote_code=True è richiesto da nomic-embed-text
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=device)
    
    print("Inizio estrazione embeddings (Batch processing)...")
    
    processed_dataset = []
    
    for graph in tqdm(dataset, desc="Vettorizzazione Grafi (Batch)"):
        # Vettorizzazione in batch dei NODI
        if hasattr(graph, 'node_text') and graph.node_text:
            # model.encode() gestisce il batching in modo ultra-efficiente su GPU
            # convert_to_tensor=True restituisce direttamente un tensore PyTorch
            graph.x = model.encode(graph.node_text, convert_to_tensor=True).cpu()
        else:
            graph.x = torch.empty((0, 768), dtype=torch.float)
            
        # Vettorizzazione in batch degli ARCHI
        if hasattr(graph, 'edge_text') and graph.edge_text:
            graph.edge_attr = model.encode(graph.edge_text, convert_to_tensor=True).cpu()
        else:
            # Se il grafo non ha archi, assegniamo un tensore vuoto
            graph.edge_attr = torch.empty((0, 768), dtype=torch.float)
            
        processed_dataset.append(graph)
        
    print(f"Salvataggio risultati in {current_output}...")
    # Salva il singolo grafo se eravamo in modalità singola, altrimenti la lista
    final_output = processed_dataset[0] if is_single_graph else processed_dataset
    
    os.makedirs(os.path.dirname(current_output), exist_ok=True)
    torch.save(final_output, current_output)
    print("✅ Fatto! Salvataggio completato con successo.")

if __name__ == "__main__":
    main()
