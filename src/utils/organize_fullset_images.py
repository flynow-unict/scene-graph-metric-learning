import os
import torch
import time

def get_ids_from_pt(pt_file):
    print(f"Loading IDs from {pt_file}...")
    data = torch.load(pt_file, map_location='cpu', weights_only=False)
    ids = set()
    for item in data:
        img_id = getattr(item, 'image_id', None)
        if img_id is None and isinstance(item, dict):
            img_id = item.get('image_id')
        if isinstance(img_id, list):
            img_id = img_id[0]
        if isinstance(img_id, torch.Tensor):
            img_id = img_id.item()
        ids.add(str(img_id))
    return ids

def main():
    start_time = time.time()
    
    script_dir = os.path.dirname(os.path.abspath(__file__)) # src/utils
    progetto_dir = os.path.dirname(os.path.dirname(script_dir)) # Progetto
    data_dir = os.path.join(progetto_dir, "data")
    
    scene_dir = os.path.join(data_dir, "sceneGraph", "raw", "full_set")
    src_images_dir = os.path.join(data_dir, "images", "images")
    target_base_dir = os.path.join(data_dir, "images", "fullset")
    
    splits_config = {
        "train": os.path.join(scene_dir, "train_scene_graphs.pt"),
        "val": os.path.join(scene_dir, "val_scene_graphs.pt"),
        "VectorDB": os.path.join(scene_dir, "test_gallery_scene_graphs.pt"),
        "Test": os.path.join(scene_dir, "test_queries_scene_graphs.pt"),
    }
    
    print(f"Source images folder: {src_images_dir}")
    print(f"Target fullset folder: {target_base_dir}")
    
    if not os.path.exists(src_images_dir):
        print(f"Error: Source directory {src_images_dir} does not exist!")
        return

    # 1. Caricamento degli ID di tutti gli split
    split_ids = {}
    all_split_ids = set()
    
    for split_name, pt_path in splits_config.items():
        ids = get_ids_from_pt(pt_path)
        split_ids[split_name] = ids
        all_split_ids.update(ids)
        print(f"  [{split_name}] Total IDs: {len(ids)}")
        
        target_dir = os.path.join(target_base_dir, split_name)
        os.makedirs(target_dir, exist_ok=True)

    print(f"\nTotal unique split IDs: {len(all_split_ids)}")
    
    # 2. Elaborazione delle immagini nella cartella sorgente
    all_files = [f for f in os.listdir(src_images_dir) if os.path.isfile(os.path.join(src_images_dir, f))]
    total_files = len(all_files)
    print(f"Total image files in source directory: {total_files}")
    
    moved_counts = {k: 0 for k in splits_config.keys()}
    missing_counts = {k: 0 for k in splits_config.keys()}
    deleted_count = 0
    
    print("\nMoving images into split directories...")
    
    for split_name, ids in split_ids.items():
        dst_folder = os.path.join(target_base_dir, split_name)
        for img_id in ids:
            filename = f"{img_id}.jpg"
            src_file = os.path.join(src_images_dir, filename)
            dst_file = os.path.join(dst_folder, filename)
            
            if os.path.exists(src_file):
                os.replace(src_file, dst_file)
                moved_counts[split_name] += 1
            elif os.path.exists(dst_file):
                moved_counts[split_name] += 1
            else:
                missing_counts[split_name] += 1
                
        print(f"  Moved to {split_name}: {moved_counts[split_name]} (Missing: {missing_counts[split_name]})")

    # 3. Rimozione delle immagini in eccesso non associate
    remaining_files = os.listdir(src_images_dir)
    print(f"\nCleaning up unassociated images ({len(remaining_files)} remaining)...")
    
    for f in remaining_files:
        filepath = os.path.join(src_images_dir, f)
        if os.path.isfile(filepath):
            os.remove(filepath)
            deleted_count += 1
            
    print(f"Deleted {deleted_count} unassociated image files.")
    
    # 4. Eliminazione della cartella sorgente ormai vuota
    if len(os.listdir(src_images_dir)) == 0:
        os.rmdir(src_images_dir)
        print(f"Removed empty source folder: {src_images_dir}")
        
    elapsed = time.time() - start_time
    print(f"\n=== SUMMARY ===")
    for split_name in splits_config.keys():
        dest = os.path.join(target_base_dir, split_name)
        cnt = len(os.listdir(dest)) if os.path.exists(dest) else 0
        print(f"  {target_base_dir}/{split_name}: {cnt} files")
    print(f"Total deleted extra files: {deleted_count}")
    print(f"Done in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
