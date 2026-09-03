import torch
import os

def create_subset_baseline():
    print("Caricamento Full Set Baseline (grafi puri)...")
    full_baseline_path = "data/sceneGraph/raw/full_set/baseline_scene_graphs.pt"
    if not os.path.exists(full_baseline_path):
        print(f"Errore: file non trovato {full_baseline_path}")
        return
        
    full_baseline = torch.load(full_baseline_path, weights_only=False)
    
    # Crea mappa da image_id a grafo per ricerca istantanea
    print(f"Indicizzazione di {len(full_baseline)} grafi del Full Set...")
    baseline_map = {str(g.image_id): g for g in full_baseline}
    
    splits = ["train_scene_graphs.pt", "val_scene_graphs.pt", "test_gallery_scene_graphs.pt", "test_queries_scene_graphs.pt"]
    
    in_dir = "data/sceneGraph/subset/raw/class_rel"
    out_dir_raw = "data/sceneGraph/subset/raw/class"
    
    os.makedirs(out_dir_raw, exist_ok=True)
    
    print("\nEstrapolazione Subset Baseline per mantenere gli stessi split di class_rel...")
    total_subset = 0
    for split in splits:
        print(f"Elaborazione {split}...")
        rel_path = os.path.join(in_dir, split)
        if not os.path.exists(rel_path):
            print(f"File {rel_path} non trovato. Salto.")
            continue
            
        rel_data = torch.load(rel_path, weights_only=False)
        
        baseline_data = []
        for g in rel_data:
            img_id = str(g.image_id)
            if img_id in baseline_map:
                baseline_data.append(baseline_map[img_id])
            else:
                print(f"ATTENZIONE: Image ID {img_id} non trovato nella baseline full set!")
                
        out_path = os.path.join(out_dir_raw, split)
        torch.save(baseline_data, out_path)
        print(f" -> Salvati {len(baseline_data)} grafi in {out_path}")
        total_subset += len(baseline_data)
        
    print(f"\nCompletato! Generati {total_subset} grafi per il Subset Baseline.")

if __name__ == "__main__":
    create_subset_baseline()
