import json
import os
import time

def main():
    t0 = time.time()
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base_dir = os.path.dirname(os.path.dirname(project_dir))
    raw_dir = os.path.join(base_dir, "Semantic Web", "Progetto")
    
    possible_img_dirs = [
        os.path.join(project_dir, "data", "images", "images"),
        os.path.join(project_dir, "data", "images", "full_set"),
        os.path.join(project_dir, "data", "images", "images_all_15k"),
        os.path.join(project_dir, "data", "images", "imagesTest"),
        os.path.join(project_dir, "data", "images", "imageVectorDB")
    ]

    print("Indicizzazione dei file presenti su disco in memoria (veloce)...")
    available_files = set()
    for d in possible_img_dirs:
        if os.path.exists(d):
            available_files.update(os.listdir(d))
    print(f"Indicizzati {len(available_files)} file totali su disco.")

    json_files = ["train_sceneGraphs.json", "val_sceneGraphs.json"]
    
    total_graphs = 0
    found_images = 0
    missing_images = 0

    for jf in json_files:
        path = os.path.join(raw_dir, jf)
        if not os.path.exists(path):
            continue
            
        print(f"Leggendo {jf} (questo puo' richiedere qualche secondo)...")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        print("Calcolo in corso...")
        for img_id in data.keys():
            total_graphs += 1
            filename = f"{img_id}.jpg"
            if filename in available_files:
                found_images += 1
            else:
                missing_images += 1

    print("\n================================")
    print(f"Totale immagini richieste dai grafi GQA (Train+Val): {total_graphs}")
    print(f"Immagini EFFETTIVAMENTE TROVATE su disco (in tutte le sottocartelle): {found_images}")
    print(f"Immagini MANCANTI: {missing_images}")
    print(f"================================\nTempo impiegato: {time.time()-t0:.1f} secondi")

if __name__ == "__main__":
    main()
