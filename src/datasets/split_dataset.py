import torch
import random
import os

script_dir = os.path.dirname(os.path.abspath(__file__)) # src/datasets
script_folder = os.path.dirname(script_dir)             # Progetto/script
project_dir = os.path.dirname(script_folder)            # Progetto
data_dir = os.path.join(project_dir, "data")

# Percorsi dei file di input nelle rispettive cartelle
input_scene = os.path.join(data_dir, "sceneGraph/Rel", "dataset_gqa_scene_graphs.pt")
input_embed = os.path.join(data_dir, "embedding/Rel", "dataset_gqa_embedded.pt")

def main():
    print(f"[*] Caricamento dataset base da {input_scene}...")
    dataset_scene = torch.load(input_scene, weights_only=False)
    
    print(f"[*] Caricamento dataset arricchito da {input_embed}...")
    dataset_embed = torch.load(input_embed, weights_only=False)
    
    assert len(dataset_scene) == len(dataset_embed), "ERRORE: I dataset hanno lunghezze diverse!"
    
    total = len(dataset_scene)
    print(f"[*] Totale grafi trovati in ciascun file: {total}")
    
    # 1. Mischiamo i due dataset IN SINCRONIA per mantenere la corrispondenza
    print("[*] Mescolamento sincronizzato dei due dataset (seed 42)...")
    combined = list(zip(dataset_scene, dataset_embed))
    random.seed(42)
    random.shuffle(combined)
    
    # Unzip dopo il mescolamento
    dataset_scene_shuffled, dataset_embed_shuffled = zip(*combined)
    dataset_scene_shuffled = list(dataset_scene_shuffled)
    dataset_embed_shuffled = list(dataset_embed_shuffled)
    
    # 2. Calcolo dimensioni
    train_size = int(0.65 * total)
    val_size = int(0.10 * total)
    test_gallery_size = int(0.20 * total)
    test_queries_size = total - train_size - val_size - test_gallery_size
    
    # Funzione helper per tagliare le fette
    def get_slices(data_list):
        return (
            data_list[:train_size],
            data_list[train_size : train_size + val_size],
            data_list[train_size + val_size : train_size + val_size + test_gallery_size],
            data_list[train_size + val_size + test_gallery_size : ]
        )
    
    # Slicing effettivo
    train_scene, val_scene, test_gallery_scene, test_queries_scene = get_slices(dataset_scene_shuffled)
    train_embed, val_embed, test_gallery_embed, test_queries_embed = get_slices(dataset_embed_shuffled)
    
    print("\n--- RISULTATO SPLIT ---")
    print(f"    Train set:                            {len(train_scene)} grafi (~65%)")
    print(f"    Validation set:                       {len(val_scene)} grafi (~10%)")
    print(f"    Test set (Gallery per FAISS):         {len(test_gallery_scene)} grafi (~20%)")
    print(f"    Test set (Queries / Inferenza Reale): {len(test_queries_scene)} grafi (~5%)")
    
    # 4. Salvataggio
    scene_dir = os.path.join(data_dir, "sceneGraph")
    embed_dir = os.path.join(data_dir, "embedding")
    
    print("\n[*] Salvataggio dei file in data/sceneGraph/ ...")
    torch.save(train_scene, os.path.join(scene_dir, "train_scene_graphs.pt"))
    torch.save(val_scene, os.path.join(scene_dir, "val_scene_graphs.pt"))
    torch.save(test_gallery_scene, os.path.join(scene_dir, "test_gallery_scene_graphs.pt"))
    torch.save(test_queries_scene, os.path.join(scene_dir, "test_queries_scene_graphs.pt"))
    
    print("[*] Salvataggio dei file in data/embedding/ ...")
    torch.save(train_embed, os.path.join(embed_dir, "train_embedded.pt"))
    torch.save(val_embed, os.path.join(embed_dir, "val_embedded.pt"))
    torch.save(test_gallery_embed, os.path.join(embed_dir, "test_gallery_embedded.pt"))
    torch.save(test_queries_embed, os.path.join(embed_dir, "test_queries_embedded.pt"))
    
    print("\n[✔] Split sincronizzato completato.")

if __name__ == "__main__":
    main()
