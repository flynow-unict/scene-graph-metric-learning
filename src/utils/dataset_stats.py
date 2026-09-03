import torch
import collections
import json
import os

def extract_stats(path, name):
    print(f"Loading {name} from {path}...")
    try:
        data = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Failed to load {name}: {e}")
        return None

    if not isinstance(data, list):
        print(f"{name} is not a list. Type: {type(data)}")
        return None

    stats = {
        "name": name,
        "num_images": len(data),
        "image_ids": [],
        "total_nodes": 0,
        "total_edges": 0,
        "empty_graphs_edges": 0,
        "node_text_counts": collections.Counter(),
        "edge_text_counts": collections.Counter()
    }

    for g in data:
        img_id = getattr(g, 'image_id', getattr(g, 'id', None))
        if img_id is not None:
            stats["image_ids"].append(str(img_id))

        # Nodi
        n_nodes = g.num_nodes if hasattr(g, 'num_nodes') else 0
        if n_nodes == 0 and hasattr(g, 'x'):
            n_nodes = g.x.shape[0]
        stats["total_nodes"] += n_nodes

        if hasattr(g, 'node_text') and g.node_text:
            for nt in g.node_text:
                # Il testo spesso è "nome. Attributes: ..."
                # Estraiamo solo il nome per le statistiche delle classi
                class_name = nt.split('.')[0].strip().lower()
                stats["node_text_counts"][class_name] += 1

        # Archi
        e = 0
        if hasattr(g, 'edge_index') and g.edge_index is not None:
            e = g.edge_index.shape[1]
        stats["total_edges"] += e
        if e == 0:
            stats["empty_graphs_edges"] += 1

        if hasattr(g, 'edge_text') and g.edge_text:
            for et in g.edge_text:
                stats["edge_text_counts"][et.strip().lower()] += 1

    stats["top_30_nodes"] = stats["node_text_counts"].most_common(30)
    stats["top_30_edges"] = stats["edge_text_counts"].most_common(30)
    stats["distinct_nodes"] = len(stats["node_text_counts"])
    stats["distinct_edges"] = len(stats["edge_text_counts"])
    
    # Remove large objects before returning
    del stats["node_text_counts"]
    del stats["edge_text_counts"]
    
    return stats

def main():
    paths = {
        "Virtuoso": "data/sceneGraph/Rel/dataset_gqa_scene_graphs.pt",
        "Dario_Full": "Dario/sceneGraph/gqa_full_scene_graphs.pt",
        "Dario_Semantic": "Dario/sceneGraph/gqa_semantic_scene_graphs.pt"
    }

    results = {}
    for name, path in paths.items():
        if os.path.exists(path):
            stats = extract_stats(path, name)
            if stats:
                results[name] = stats
        else:
            print(f"File not found: {path}")

    # Compare image IDs
    if "Virtuoso" in results and "Dario_Full" in results:
        set_virt = set(results["Virtuoso"]["image_ids"])
        set_dario = set(results["Dario_Full"]["image_ids"])
        
        results["Comparison"] = {
            "virtuoso_only_ids": len(set_virt - set_dario),
            "dario_only_ids": len(set_dario - set_virt),
            "common_ids": len(set_virt.intersection(set_dario))
        }
        
    for k in results.keys():
        if "image_ids" in results[k]:
            del results[k]["image_ids"] # clean up huge lists

    with open("dataset_comparison_stats.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Done. Saved to dataset_comparison_stats.json")

if __name__ == "__main__":
    main()
