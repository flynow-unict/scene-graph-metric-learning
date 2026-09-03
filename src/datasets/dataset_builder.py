from src.semantic_web.client import VirtuosoClient
from src.semantic_web.query import GQAQueryFactory
from src.utils import paths
import torch
from torch_geometric.data import Data
import argparse

def main():
    parser = argparse.ArgumentParser(description="Genera dataset da Virtuoso")
    parser.add_argument("--baseline", action="store_true", help="Genera dati puri GQA senza arricchimento semantico")
    args = parser.parse_args()

    VC = VirtuosoClient()

    qCountImages = VC.execute_query_json(GQAQueryFactory.count_images())

    num_img = qCountImages[0]['num_img']
    print(f"Analizzo {num_img} immagini...")

    qSceneIds = VC.execute_query_json(GQAQueryFactory.get_scene_ids(limit = None))
    qSceneIdsFormated = [int(img['img'].replace("http://progetto-dl-sw.org/images/","")) for img in qSceneIds]
    qSceneIdsFormated.sort()

    zero_rel_img = []
    zero_obj_img = []
    no_reasoning_images = []
    dataset_list = []

    for idx, scene_id in enumerate(qSceneIdsFormated):
        print(f"Immagine {idx+1}/{num_img} -> Id: {scene_id}")    

        qObjects = VC.execute_query_json(GQAQueryFactory.image_objects_and_attributes(scene_id))
        n_obj = len(qObjects)
        for obj in qObjects:
            obj['classeBase'] = str(obj['classeBase']).replace("http://progetto-dl-sw.org/ontology#","")
            obj['oggetto'] = str(obj['oggetto']).replace("http://progetto-dl-sw.org/objects/","")
            obj['macroClassi'] = str(obj['macroClassi']).replace("http://example.invalid/ontologies/ont.owl#","")
            obj['attributi'] = str(obj['attributi']).replace("http://progetto-dl-sw.org/ontology#","")

        # Prova la query con ragionatore delegando il timeout di 10s nativamente a urllib/SPARQLWrapper
        if args.baseline:
            query_str = GQAQueryFactory.image_relations_classes(scene_id, reasoning=False)
        else:
            query_str = GQAQueryFactory.image_relations_classes(scene_id, reasoning=True)
        
        qRelations = VC.execute_query_json(query_str, timeout=1)
    
        # Se va in timeout, restituisce None. Allora proviamo senza ragionatore.
        if qRelations is None:
            print(f"[!] Timeout (1s) superato per immagine {scene_id}, rilancio senza ragionatore...")
            no_reasoning_images.append(scene_id)
            query_str_no_reasoning = GQAQueryFactory.image_relations_classes(scene_id, reasoning=False)
            qRelations = VC.execute_query_json(query_str_no_reasoning)
            if qRelations is None:
                qRelations = [] # Fallback di sicurezza estrema
            
        n_rel = len(qRelations)
        for rel in qRelations:
            rel['soggetto'] = str(rel['soggetto']).replace("http://progetto-dl-sw.org/objects/","")
            rel['oggettoDestinazione'] = str(rel['oggettoDestinazione']).replace("http://progetto-dl-sw.org/objects/","")
            rel['proprieta'] = str(rel['proprieta']).replace("http://progetto-dl-sw.org/ontology#","")
        print(f"| N_obj: {n_obj} | N_rel: {n_rel}")
        if n_obj == 0:
            zero_obj_img.append(scene_id)
            continue # Se non ha oggetti, non ha senso creare il grafo
        if n_rel == 0:
            zero_rel_img.append(scene_id)

        # --- CREAZIONE DEL GRAFO PYTORCH GEOMETRIC ---
        node_uri_to_idx = {}
        node_features_text = []
    
        for idx_node, obj in enumerate(qObjects):
            node_uri_to_idx[obj['oggetto']] = idx_node
        
            classe = obj.get('classeBase', '')
            macro = obj.get('macroClassi', '')
            attr = obj.get('attributi', '')
        
            if args.baseline:
                testo_nomic = f"{classe}. Attributes: {attr}."
            else:
                testo_nomic = f"{classe}. Categories: {macro}. Attributes: {attr}."
            
            node_features_text.append(testo_nomic)
        
        num_nodes = len(node_features_text)
        # Vettore fittizio temporaneo per x (da sostituire con gli embedding NOMIC)
        x = torch.randn((num_nodes, 768), dtype=torch.float)
    
        source_nodes = []
        target_nodes = []
        edge_features_text = []
    
        for rel in qRelations:
            sogg = rel['soggetto']
            ogg = rel['oggettoDestinazione']
            proprieta = str(rel['proprieta']).replace("http://progetto-dl-sw.org/ontology#", "")
            # PyTorch richiede ID continui (0, 1, 2...). Usiamo la mappa
            if sogg in node_uri_to_idx and ogg in node_uri_to_idx:
                source_nodes.append(node_uri_to_idx[sogg])
                target_nodes.append(node_uri_to_idx[ogg])
                edge_features_text.append(proprieta)
            
        if len(source_nodes) > 0:
            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        graph = Data(x=x, edge_index=edge_index)
        graph.image_id = scene_id
        graph.node_text = node_features_text # Memorizziamo il testo per vettorizzarlo poi
        graph.edge_text = edge_features_text # Memorizziamo la relazione testuale
    
        dataset_list.append(graph)

    print(f"\n{len(zero_rel_img)} Immagini con zero relazioni: {zero_rel_img}")
    print(f"\n{len(zero_obj_img)} Immagini con zero oggetti: {zero_obj_img}")
    print(f"\n{len(no_reasoning_images)} Immagini con timeout: {no_reasoning_images}")

    # --- SALVATAGGIO ---
    if args.baseline:
        save_path = paths.SCENE_GRAPH_DIR / "dataset_gqa_scene_graphs_baseline.pt"
    else:
        save_path = paths.SCENE_GRAPH_DIR / "dataset_gqa_scene_graphs.pt"
        
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset_list, save_path)
    print(f"\n[✔] Dataset salvato in {save_path} ({len(dataset_list)} grafi validi).")

if __name__ == "__main__":
    main()



