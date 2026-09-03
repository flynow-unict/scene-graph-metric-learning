import torch
import os
import json
from tqdm import tqdm

def export_graphs():
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_dir, "data")
    
    # Del fullset si esporta il solo test_queries: la gallery pesa 6.5 GB e non serve al
    # frontend, che confronta le query contro la gallery del subset.
    
    files_to_export = [
        os.path.join(data_dir, "sceneGraph", "fullset", "semantic", "embedded", "test_queries_scene_graphs.pt"),
        os.path.join(data_dir, "sceneGraph", "subset", "semantic", "embedded", "test_queries_scene_graphs.pt"),
        os.path.join(data_dir, "sceneGraph", "subset", "semantic", "embedded", "test_gallery_scene_graphs.pt"),
    ]
    
    count = 0
    for file_path in files_to_export:
        if not os.path.exists(file_path):
            print(f"File non trovato: {file_path}")
            continue
            
        print(f"Caricamento {file_path} ...")
        dataset = torch.load(file_path, map_location='cpu', weights_only=False)
        
        for graph in tqdm(dataset, desc=f"Esportazione {os.path.basename(file_path)}"):
            image_id = str(getattr(graph, 'image_id', '')).strip()
            if not image_id:
                continue
                
            graph_json = {
                "nodes": [{"id": str(i), "label": text} for i, text in enumerate(getattr(graph, 'node_text', []))],
                "edges": []
            }
            if hasattr(graph, 'edge_index') and graph.edge_index is not None and graph.edge_index.numel() > 0:
                edge_index = graph.edge_index.tolist()
                edge_text = getattr(graph, 'edge_text', [])
                for i in range(len(edge_index[0])):
                    graph_json["edges"].append({
                        "source": str(edge_index[0][i]),
                        "target": str(edge_index[1][i]),
                        "label": edge_text[i] if i < len(edge_text) else ""
                    })
                    
            out_dir = os.path.join(data_dir, "sceneGraph", "json", "subset" if "subset" in file_path else "fullset")
            os.makedirs(out_dir, exist_ok=True)
            json_path = os.path.join(out_dir, f"{image_id}.json")
            with open(json_path, 'w') as f:
                json.dump(graph_json, f)
            count += 1
            
    print(f"Completato! Esportati {count} grafi in {out_dir}")

if __name__ == "__main__":
    export_graphs()
