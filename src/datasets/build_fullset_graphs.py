import torch
from torch_geometric.data import Data
import json
import os
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# --- REGOLE ONTOLOGICHE ---

SYMMETRIC_PROPERTIES = {
    'near', 'next_to', 'beside', 'by', 'around', 'touching', 'with', 
    'standing_next_to', 'standing_by'
}

INVERSE_PROPERTIES = {
    'above': 'below',
    'below': 'above',
    'under': 'above',
    'in_front_of': 'behind',
    'behind': 'in_front_of',
    'to_the_left_of': 'to_the_right_of',
    'to_the_right_of': 'to_the_left_of',
    'on_the_front_of': 'on_the_back_of',
    'on_the_back_of': 'on_the_front_of',
    'covering': 'covered_by',
    'covered_by': 'covering',
    'pulling': 'pulled_by',
    'pulled_by': 'pulling',
    'contain': 'in',
    'in': 'contain',
    'inside': 'contain'
}

SUB_PROPERTIES = {
    'next_to': 'near',
    'beside': 'near',
    'by': 'near',
    'around': 'near',
    'standing_next_to': 'near',
    'standing_by': 'near',
    'inside': 'in',
    'sitting_in': 'in',
    'standing_in': 'in',
    'lying_in': 'in',
    'flying_in': 'in',
    'walking_in': 'in',
    'on_top_of': 'on',
    'sitting_on': 'on',
    'standing_on': 'on',
    'lying_on': 'on',
    'riding_on': 'on',
    'parked_on': 'on',
    'walking_on': 'on',
    'driving_on': 'on',
    'leaning_on': 'touching',
    'hitting': 'touching'
}

TRANSITIVE_PROPERTIES = {
    'in', 'inside', 'contain', 'above', 'below', 'under', 'behind', 
    'in_front_of', 'to_the_left_of', 'to_the_right_of', 'hanging_from', 
    'covering', 'covered_by', 'pulling', 'pulled_by'
}

# (prop1, prop2) -> proprietà inferita
PROPERTY_CHAINS = {
    ('sitting_on', 'in'): 'in',
    ('standing_on', 'in'): 'in',
    ('lying_on', 'in'): 'in',
    ('parked_on', 'in'): 'in',
    ('walking_in', 'in'): 'in',
    ('flying_in', 'in'): 'in',
    
    ('sitting_on', 'on'): 'above',
    ('standing_on', 'on'): 'above',
    ('parked_on', 'on'): 'above',
    ('lying_on', 'on'): 'above',
    ('walking_on', 'on'): 'above',
    
    ('in', 'near'): 'near',
    ('on', 'near'): 'near',
    
    ('in', 'behind'): 'behind',
    ('on', 'behind'): 'behind',
    
    ('in', 'in_front_of'): 'in_front_of',
    ('on', 'in_front_of'): 'in_front_of',
    
    ('riding', 'in_front_of'): 'looking_at',
    ('riding_on', 'in_front_of'): 'looking_at',
    
    ('wearing', 'covered_in'): 'covered_in',
    ('wearing', 'covered_with'): 'covered_with'
}


def apply_reasoning(logical_edges):
    """
    Applies logic rules to the set of logical_edges.
    logical_edges is a set of tuples: (source_idx, target_idx, property_name)
    Returns the expanded set of logical edges.
    """
    added_new = True
    
    while added_new:
        added_new = False
        new_edges = set()
        
        # Indice degli archi per risalire velocemente a catene e transitività
        # dict: sorgente -> lista di (destinazione, proprietà)
        adj = {}
        for u, v, p in logical_edges:
            if u not in adj:
                adj[u] = []
            adj[u].append((v, p))
            
        for u, v, p in logical_edges:
            # 1. Simmetria
            if p in SYMMETRIC_PROPERTIES:
                if (v, u, p) not in logical_edges and (v, u, p) not in new_edges:
                    new_edges.add((v, u, p))
            
            # 2. Proprietà inverse
            if p in INVERSE_PROPERTIES:
                inv_p = INVERSE_PROPERTIES[p]
                if (v, u, inv_p) not in logical_edges and (v, u, inv_p) not in new_edges:
                    new_edges.add((v, u, inv_p))
                    
            # 3. Sotto-proprietà
            if p in SUB_PROPERTIES:
                sup_p = SUB_PROPERTIES[p]
                if (u, v, sup_p) not in logical_edges and (u, v, sup_p) not in new_edges:
                    new_edges.add((u, v, sup_p))
                    
            # 4 e 5. Transitività e catene di proprietà
            if v in adj:
                for w, p2 in adj[v]:
                    # Transitività (p o p -> p)
                    if p == p2 and p in TRANSITIVE_PROPERTIES:
                        if (u, w, p) not in logical_edges and (u, w, p) not in new_edges and u != w:
                            new_edges.add((u, w, p))
                            
                    # Catene di proprietà (p1 o p2 -> p3)
                    if (p, p2) in PROPERTY_CHAINS:
                        res_p = PROPERTY_CHAINS[(p, p2)]
                        if (u, w, res_p) not in logical_edges and (u, w, res_p) not in new_edges and u != w:
                            new_edges.add((u, w, res_p))
                            
        if new_edges:
            logical_edges.update(new_edges)
            added_new = True
            
    return logical_edges


def main():
    parser = argparse.ArgumentParser(description="Genera i grafi di GQA.")
    parser.add_argument("--baseline", action="store_true", help="Non applicare il ragionamento logico (Ablation).")
    args = parser.parse_args()

    print("Loading sentence-transformers model...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Error loading sentence-transformers. Please pip install sentence-transformers: {e}")
        return

    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_dir, "data")
    subsample_path = os.path.join(data_dir, "sceneGraph", "subset", "raw", "class_rel", "dataset_gqa_scene_graphs.pt")
    
    semantic_web_dir = os.path.abspath(os.path.join(project_dir, "..", "..", "Semantic Web"))
    gqa_train_json = os.path.join(semantic_web_dir, "Progetto", "train_sceneGraphs.json")
    gqa_val_json = os.path.join(semantic_web_dir, "Progetto", "val_sceneGraphs.json")
    gqa_train_json_2 = os.path.join(semantic_web_dir, "consegna2", "train_sceneGraphs.json")
    gqa_val_json_2 = os.path.join(semantic_web_dir, "consegna2", "val_sceneGraphs.json")
    
    json_paths = [gqa_train_json, gqa_val_json, gqa_train_json_2, gqa_val_json_2]
    valid_jsons = []
    for jp in json_paths:
        if os.path.exists(jp) and os.path.basename(jp) not in [os.path.basename(v) for v in valid_jsons]:
            valid_jsons.append(jp)

    print(f"Loading subsample to extract mappings from {subsample_path}...")
    try:
        subsample = torch.load(subsample_path, map_location='cpu', weights_only=False)
    except FileNotFoundError:
        print(f"Subsample not found at {subsample_path}. Make sure the path is correct.")
        return

    class_to_macro = {}
    known_properties = set()

    for data in subsample:
        if hasattr(data, 'node_text'):
            for text in data.node_text:
                parts = text.split('. Categories: ')
                if len(parts) == 2:
                    cls = parts[0].strip()
                    rest = parts[1].split('. Attributes: ')
                    if len(rest) == 2:
                        macro = rest[0].strip()
                        class_to_macro[cls] = macro
        if hasattr(data, 'edge_text'):
            for e_text in data.edge_text:
                known_properties.add(e_text.strip())

    print(f"Extracted mappings for {len(class_to_macro)} unique classes.")
    known_classes = list(class_to_macro.keys())
    known_properties = list(known_properties)
    print(f"Extracted {len(known_properties)} unique properties.")
        
    print("Pre-computing embeddings for known classes and properties...")
    known_embeddings = model.encode(known_classes, convert_to_tensor=True)
    known_prop_embeddings = model.encode(known_properties, convert_to_tensor=True)

    inference_cache = {}
    prop_inference_cache = {}

    def get_macro_for_class(cls_name):
        if cls_name in class_to_macro:
            return class_to_macro[cls_name]
        if cls_name in inference_cache:
            return inference_cache[cls_name]

        cls_emb = model.encode([cls_name], convert_to_tensor=True)
        cos_scores = util.cos_sim(cls_emb, known_embeddings)[0]
        best_idx = torch.argmax(cos_scores).item()
        best_class = known_classes[best_idx]
        inferred_macro = class_to_macro[best_class]
        
        inference_cache[cls_name] = inferred_macro
        return inferred_macro

    def get_property_mapping(raw_prop):
        if raw_prop in known_properties:
            return raw_prop
        if raw_prop in prop_inference_cache:
            return prop_inference_cache[raw_prop]
            
        prop_emb = model.encode([raw_prop], convert_to_tensor=True)
        cos_scores = util.cos_sim(prop_emb, known_prop_embeddings)[0]
        best_idx = torch.argmax(cos_scores).item()
        best_prop = known_properties[best_idx]
        
        prop_inference_cache[raw_prop] = best_prop
        return best_prop

    all_graphs = []
    
    for json_path in valid_jsons:
        print(f"Processing GQA file: {json_path}")
        with open(json_path, 'r') as f:
            gqa_data = json.load(f)

        for img_id, scene_data in tqdm(gqa_data.items(), desc=f"Parsing {os.path.basename(json_path)}"):
            objects = scene_data.get('objects', {})
            
            if len(objects) == 0:
                continue

            obj_id_to_idx = {}
            node_features_text = []
            
            sorted_obj_ids = sorted(list(objects.keys()))
            for idx, obj_id in enumerate(sorted_obj_ids):
                obj_id_to_idx[obj_id] = idx
                obj_info = objects[obj_id]
                
                cls_name = obj_info.get('name', 'unknown')
                attrs = obj_info.get('attributes', [])
                
                attr_str = ", ".join(attrs) if attrs else ""
                
                if args.baseline:
                    testo_nomic = f"{cls_name}. Attributes: {attr_str}."
                else:
                    macro = get_macro_for_class(cls_name)
                    testo_nomic = f"{cls_name}. Categories: {macro}. Attributes: {attr_str}."
                node_features_text.append(testo_nomic)
                
            num_nodes = len(node_features_text)
            x = torch.randn((num_nodes, 768), dtype=torch.float)
            
            # Estrazione degli archi
            raw_edges = []
            logical_edges = set()
            mapped_raw_edges_set = set()
            
            for obj_id in sorted_obj_ids:
                obj_info = objects[obj_id]
                relations = obj_info.get('relations', [])
                for rel in relations:
                    target_id = rel.get('object')
                    rel_name = rel.get('name', 'related_to')
                    
                    if target_id in obj_id_to_idx:
                        u = obj_id_to_idx[obj_id]
                        v = obj_id_to_idx[target_id]
                        raw_edges.append((u, v, rel_name))
                        
                        mapped_rel_name = get_property_mapping(rel_name)
                        logical_edges.add((u, v, mapped_rel_name))
                        mapped_raw_edges_set.add((u, v, mapped_rel_name))
            
            # Ragionamento logico applicato solo se non è la variante baseline
            if not args.baseline:
                expanded_logical_edges = apply_reasoning(logical_edges)
            else:
                expanded_logical_edges = set()
            
            final_source_nodes = []
            final_target_nodes = []
            final_edge_features_text = []
            
            # Tutti gli archi raw
            for u, v, rel_name in raw_edges:
                final_source_nodes.append(u)
                final_target_nodes.append(v)
                final_edge_features_text.append(rel_name)
                
            # Solo i nuovi archi inferiti, non quelli già derivati dai raw
            if not args.baseline:
                for u, v, logic_p in expanded_logical_edges:
                    if (u, v, logic_p) not in mapped_raw_edges_set:
                        final_source_nodes.append(u)
                        final_target_nodes.append(v)
                        final_edge_features_text.append(logic_p)
            
            if len(final_source_nodes) > 0:
                edge_index = torch.tensor([final_source_nodes, final_target_nodes], dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                
            graph = Data(x=x, edge_index=edge_index)
            graph.image_id = img_id
            graph.node_text = node_features_text
            graph.edge_text = final_edge_features_text
            
            all_graphs.append(graph)

    out_dir = os.path.join(data_dir, "sceneGraph", "raw", "full_set")
    os.makedirs(out_dir, exist_ok=True)
    
    filename = "baseline_scene_graphs.pt" if args.baseline else "full_scene_graphs.pt"
    out_path = os.path.join(out_dir, filename)
    
    print(f"Saving {len(all_graphs)} graphs to {out_path}...")
    torch.save(all_graphs, out_path)
    print("Done!")

if __name__ == "__main__":
    main()
