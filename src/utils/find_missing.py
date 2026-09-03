import json
import os

def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base_dir = os.path.dirname(os.path.dirname(project_dir))
    raw_dir = os.path.join(base_dir, "Semantic Web", "Progetto")
    images_dir = os.path.join(project_dir, "data", "images", "images")

    json_files = ["train_sceneGraphs.json", "val_sceneGraphs.json"]
    
    missing_ids = []

    for jf in json_files:
        path = os.path.join(raw_dir, jf)
        if not os.path.exists(path):
            continue
            
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for img_id in data.keys():
            img_path = os.path.join(images_dir, f"{img_id}.jpg")
            if not os.path.exists(img_path):
                missing_ids.append(img_id)
                if len(missing_ids) >= 10:
                    break
        if len(missing_ids) >= 10:
            break

    print("Missing IDs:")
    for mid in missing_ids:
        print(mid)

if __name__ == "__main__":
    main()
