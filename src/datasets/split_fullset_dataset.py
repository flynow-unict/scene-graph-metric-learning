import torch
import random
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Split dataset GQA.")
    parser.add_argument("--baseline", action="store_true", help="Usa il file baseline e salva solo il test set in una cartella separata.")
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_dir, "data")
    
    filename = "baseline_scene_graphs.pt" if args.baseline else "full_scene_graphs.pt"
    input_scene = os.path.join(data_dir, "sceneGraph", "raw", "full_set", filename)
    
    print(f"Loading full dataset from {input_scene}...")
    try:
        dataset = torch.load(input_scene, weights_only=False)
    except FileNotFoundError:
        print(f"Error: {input_scene} not found.")
        return
        
    total = len(dataset)
    print(f"Total graphs loaded: {total}")
    
    if total < 1000:
        print("Error: Dataset has less than 1000 graphs.")
        return
        
    print("Shuffling dataset (seed 42)...")
    random.seed(42)
    random.shuffle(dataset)
    
    # Vincolo: test queries esattamente 1000
    test_queries_size = 1000
    remaining = total - test_queries_size
    
    # Ripartizione del resto: ~65% train, ~10% val, ~25% test gallery
    train_size = int(0.65 * total)
    val_size = int(0.10 * total)
    test_gallery_size = remaining - train_size - val_size
    
    print(f"Total: {total}")
    print(f"Train size: {train_size}")
    print(f"Val size: {val_size}")
    print(f"Test Gallery size: {test_gallery_size}")
    print(f"Test Queries size: {test_queries_size}")
    
    train_scene = dataset[:train_size]
    val_scene = dataset[train_size : train_size + val_size]
    test_gallery_scene = dataset[train_size + val_size : train_size + val_size + test_gallery_size]
    test_queries_scene = dataset[train_size + val_size + test_gallery_size:]
    
    assert len(test_queries_scene) == 1000, f"Test queries size mismatch: {len(test_queries_scene)}"
    assert len(train_scene) + len(val_scene) + len(test_gallery_scene) + len(test_queries_scene) == total
    
    if args.baseline:
        out_dir_raw = os.path.join(data_dir, "sceneGraph", "raw", "full_set_baseline")
    else:
        out_dir_raw = os.path.join(data_dir, "sceneGraph", "raw", "full_set")
        
    os.makedirs(out_dir_raw, exist_ok=True)
    
    print(f"\nSaving splits to {out_dir_raw} ...")
    torch.save(train_scene, os.path.join(out_dir_raw, "train_scene_graphs.pt"))
    torch.save(val_scene, os.path.join(out_dir_raw, "val_scene_graphs.pt"))
    torch.save(test_gallery_scene, os.path.join(out_dir_raw, "test_gallery_scene_graphs.pt"))
    torch.save(test_queries_scene, os.path.join(out_dir_raw, "test_queries_scene_graphs.pt"))
    
    print("\n[✔] Split generati.")

if __name__ == "__main__":
    main()
