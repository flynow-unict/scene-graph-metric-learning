import os
import sys
import json
import base64
import argparse
import re
import requests
from PIL import Image
import torch
from torch_geometric.data import Data

try:
    from sentence_transformers import SentenceTransformer, util
    ST_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    print("Warning: sentence-transformers non trovato. Esegui pip install sentence-transformers.")
    ST_MODEL = None

REPRESENTATIVE_CLASSES = {
    "Animals, LivingBeing": ["bird", "horse", "chicken", "duck", "dog", "cat", "cow", "sheep", "elephant", "bear", "giraffe", "zebra", "animal"],
    "Beverages, Substance": ["water", "bottle", "cup", "wine", "mug", "wineglass", "drink", "beer", "juice", "beverage"],
    "Bodypart, LivingBeing": ["head", "hand", "hair", "leg", "ear", "neck", "mouth", "foot", "eye", "face", "finger", "nose", "paw", "tail", "wing", "wrist", "bodypart", "hands"],
    "Clothing, InanimateObject": ["shirt", "pants", "jacket", "hat", "shoe", "ring", "goggles", "boot", "vest", "suit", "coat", "cap", "glasses", "glove", "helmet", "jeans", "shorts", "sock", "tie", "clothing", "dress", "shield"],
    "Food, Substance": ["banana", "pizza", "donut", "cheese", "broccoli", "strawberry", "basil", "corn", "soup", "cookie", "apple", "bread", "cake", "carrot", "fruit", "meat", "orange", "tomato", "food", "sushi", "roll", "meal", "dish", "sauce", "pancake", "sausage", "steak", "burger"],
    "Furniture, InanimateObject": ["table", "chair", "mirror", "bench", "shelf", "curtain", "tent", "oven", "tablecloth", "bathtub", "cabinet", "coffeetable", "couch", "desk", "drawer", "refrigerator", "furniture", "bed", "sofa"],
    "InanimateObject, Item": ["sign", "pole", "fence", "wheel", "letter", "bucket", "symbol", "propeller", "lightswitch", "poster", "bag", "chain", "logo", "number", "picture", "post", "tire", "umbrella", "item", "plate", "bowl", "glass", "chopsticks", "fork", "knife", "spoon", "board", "tray", "box", "streetlight", "traffic signal", "clock"],
    "LivingBeing, Nature": ["tree", "ground", "sky", "grass", "leaf", "mountain", "branch", "stone", "bush", "ocean", "cloud", "field", "flower", "hill", "plant", "rock", "snow", "nature", "landscape", "dirt", "sand", "hay"],
    "LivingBeing, Person": ["man", "woman", "people", "boy", "girl", "skier", "men", "crowd", "spectator", "baby", "child", "couple", "guy", "lady", "player", "person", "rider"],
    "ArchitecturalStructure, Location": ["building", "house", "bridge", "balcony", "kitchen", "buildings", "tower", "castle", "monument", "cross"],
    "FacilityElement, Location": ["window", "wall", "floor", "pavement", "outlet", "ceiling", "door", "roof", "sink", "staircase", "runway", "windows"],
    "Location, OutdoorSpace": ["road", "sidewalk", "street", "dock", "parkinglot", "yard", "park", "avenue", "statue", "champs-élysées", "arc de triomphe"],
    "Location, IndoorSpace": ["room", "hallway", "corridor"],
    "SportsEquipment, InanimateObject": ["racket", "bike", "surfboard", "basket", "ski", "tennisball", "baseball", "baseballbat", "paddle", "bicycle", "skateboard", "snowboard", "sport"],
    "InanimateObject, Vehicles": ["car", "boat", "bus", "train", "airplane", "cart", "cockpit", "motorcycle", "truck", "vehicle", "van", "jet", "plane", "ship", "wagon"]
}

FLAT_KNOWN_CLASSES = []
FLAT_MACRO_MAPPING = []
for macro, classes in REPRESENTATIVE_CLASSES.items():
    for cls in classes:
        FLAT_KNOWN_CLASSES.append(cls)
        FLAT_MACRO_MAPPING.append(macro)

if ST_MODEL is not None:
    KNOWN_EMBEDDINGS = ST_MODEL.encode(FLAT_KNOWN_CLASSES, convert_to_tensor=True)

PROMPT = """You are an expert AI vision system. Analyze the image and extract a highly detailed semantic Scene Graph.
You MUST return ONLY a valid JSON object. Do not output flat lists of strings. You must output an array of dictionaries for objects and relations.

REQUIRED JSON STRUCTURE:
{
  "objects": [
    {"id": "node_1", "name": "man", "attributes": ["wearing red shirt", "standing", "tall", "smiling", "holding leash"]},
    {"id": "node_2", "name": "dog", "attributes": ["brown", "jumping", "small size", "fluffy", "happy"]}
  ],
  "relations": [
    {"source": "node_1", "target": "node_2", "relation": "playing with"}
  ]
}

IMPORTANT CONSTRAINTS:
1. "objects" MUST be a list of dictionaries. Each dictionary MUST have "id", "name", and a list of exactly 5 descriptive string "attributes".
2. "relations" MUST be a list of dictionaries. Each dictionary MUST have "source", "target", and "relation".
3. CAREFULLY ANALYZE the entire image. You MUST identify AT LEAST 10 (ten) diverse and prominent objects. DO NOT list fewer than 10 objects.
   Focus especially on identifying the correct types of objects (e.g. differentiate skis, ski poles, skates, surfboards, skateboards, and snowboards; or motorcycles from standard bicycles).
4. Extract AT LEAST 12 to 15 meaningful relations between these objects. Output MUST start with { and end with }.
5. CRITICAL: For the "relation" field, you MUST USE ONLY simple physical/spatial properties. Choose ONLY from this list: [on, in, near, behind, in_front_of, next_to, under, above, holding, wearing, touching, looking_at, riding, standing_on, sitting_on, parked_on, attached_to, part_of, covered_by, pulling]. 
   DO NOT invent complex or poetic relations like 'creating atmosphere' or 'inspiring admiration'. Keep them strictly spatial or physical.
6. CRITICAL: For the "name" field, you MUST use highly specific, concrete nouns that precisely describe the object (e.g., use "skier", "goggles", "ski pole", "ski lift", "snowboard", "helmet", "mountain" instead of generic words like "person", "object", "item", "tool", "equipment", "accessory", "landscape", "environment", "terrain"). Never output generic placeholders as the object name.
"""

def encode_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_json_from_response(text):
    """Estrae e pulisce il JSON se il VLM ha aggiunto markdown o testo extra."""
    text = text.replace(r'\_', '_')
    try:
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            return json.loads(text[start_idx:end_idx+1])
            
        return json.loads(text)
    except Exception as e:
        print(f"Errore nel parsing del JSON restituito dal VLM: {e}")
        print(f"Risposta raw:\n{text}")
        return None

def extract_graph_ollama(image_path, model_name="llava:latest"):
    print(f"[Ollama] Estrazione grafo da {image_path} usando {model_name}...")
    
    url = "http://localhost:11434/api/generate"
    b64_img = encode_image_base64(image_path)
    
    payload = {
        "model": model_name,
        "prompt": PROMPT,
        "images": [b64_img],
        "stream": False,
        "format": "json"
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result_text = response.json().get("response", "")
        return parse_json_from_response(result_text)
    except Exception as e:
        print(f"[Ollama] Errore di connessione: {e}")
        return None

def extract_graph_hf(image_path, model_id="llava-hf/llava-1.5-13b-hf"):
    print(f"[HuggingFace] Caricamento modello {model_id} su GPU in 4-bit...")
    try:
        from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
        import torch
    except ImportError:
        print("Librerie mancanti! Su cluster esegui: pip install transformers accelerate torch pillow bitsandbytes qwen-vl-utils")
        sys.exit(1)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[HuggingFace] Utilizzo device: {device}")
    
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"[HuggingFace] Avviso rete: {e}. Uso della cache locale...")
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
    
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    image = Image.open(image_path).convert("RGB")
    
    if "mistral" in model_id.lower() or "llava-v1.6" in model_id.lower():
        from transformers import LlavaNextForConditionalGeneration
        ModelClass = LlavaNextForConditionalGeneration
    elif "qwen" in model_id.lower():
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            ModelClass = Qwen2_5_VLForConditionalGeneration
        except ImportError:
            from transformers import Qwen2VLForConditionalGeneration
            ModelClass = Qwen2VLForConditionalGeneration
    else:
        from transformers import LlavaForConditionalGeneration
        ModelClass = LlavaForConditionalGeneration
        
    try:
        model = ModelClass.from_pretrained(
            model_id, quantization_config=quantization_config, device_map="auto",
            low_cpu_mem_usage=True, trust_remote_code=True
        )
    except Exception as e:
        print(f"[HuggingFace] Avviso rete modello: {e}. Uso cache locale...")
        model = ModelClass.from_pretrained(
            model_id, quantization_config=quantization_config, device_map="auto",
            low_cpu_mem_usage=True, trust_remote_code=True, local_files_only=True
        )
    
    if "qwen" in model_id.lower():
        from qwen_vl_utils import process_vision_info
        messages = [{"role": "user", "content": [{"type": "image", "image": f"file://{os.path.abspath(image_path)}"}, {"type": "text", "text": PROMPT}]}]
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text_prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
    else:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        try:
            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except ValueError:
            print("[HuggingFace] Nessun chat template trovato. Uso manuale...")
            if "mistral" in model_id.lower():
                text_prompt = f"[INST] <image>\n{PROMPT} [/INST]"
            else:
                text_prompt = f"USER: <image>\n{PROMPT}\nASSISTANT:"
        text_prompt += " {"
        inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to(device)
    
    print("[HuggingFace] Generazione in corso...")
    with torch.no_grad():
        if hasattr(model, "_validate_model_kwargs"):
            model._validate_model_kwargs = lambda *args, **kwargs: None
        output = model.generate(**inputs, max_new_tokens=2048, do_sample=True, temperature=0.4, top_p=0.9, repetition_penalty=1.15)
        
    decoded_output = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    if "qwen" not in model_id.lower():
        decoded_output = "{" + decoded_output
    
    return parse_json_from_response(decoded_output)

# --- REGOLE ONTOLOGICHE E RAGIONAMENTO PER L'ARRICCHIMENTO VLM ---
SYMMETRIC_PROPERTIES = {'near', 'next_to', 'beside', 'by', 'around', 'touching', 'with', 'standing_next_to', 'standing_by'}
INVERSE_PROPERTIES = {'above': 'below', 'below': 'above', 'under': 'above', 'in_front_of': 'behind', 'behind': 'in_front_of', 'to_the_left_of': 'to_the_right_of', 'to_the_right_of': 'to_the_left_of', 'on_the_front_of': 'on_the_back_of', 'on_the_back_of': 'on_the_front_of', 'covering': 'covered_by', 'covered_by': 'covering', 'pulling': 'pulled_by', 'pulled_by': 'pulling', 'contain': 'in', 'in': 'contain', 'inside': 'contain'}
SUB_PROPERTIES = {'next_to': 'near', 'beside': 'near', 'by': 'near', 'around': 'near', 'standing_next_to': 'near', 'standing_by': 'near', 'inside': 'in', 'sitting_in': 'in', 'standing_in': 'in', 'lying_in': 'in', 'flying_in': 'in', 'walking_in': 'in', 'on_top_of': 'on', 'sitting_on': 'on', 'standing_on': 'on', 'lying_on': 'on', 'riding_on': 'on', 'parked_on': 'on', 'walking_on': 'on', 'driving_on': 'on', 'leaning_on': 'touching', 'hitting': 'touching'}
TRANSITIVE_PROPERTIES = {'in', 'inside', 'contain', 'above', 'below', 'under', 'behind', 'in_front_of', 'to_the_left_of', 'to_the_right_of', 'hanging_from', 'covering', 'covered_by', 'pulling', 'pulled_by'}
PROPERTY_CHAINS = {('sitting_on', 'in'): 'in', ('standing_on', 'in'): 'in', ('lying_on', 'in'): 'in', ('parked_on', 'in'): 'in', ('walking_in', 'in'): 'in', ('flying_in', 'in'): 'in', ('sitting_on', 'on'): 'above', ('standing_on', 'on'): 'above', ('parked_on', 'on'): 'above', ('lying_on', 'on'): 'above', ('walking_on', 'on'): 'above', ('in', 'near'): 'near', ('on', 'near'): 'near', ('in', 'behind'): 'behind', ('on', 'behind'): 'behind', ('in', 'in_front_of'): 'in_front_of', ('on', 'in_front_of'): 'in_front_of', ('riding', 'in_front_of'): 'looking_at', ('riding_on', 'in_front_of'): 'looking_at', ('wearing', 'covered_in'): 'covered_in', ('wearing', 'covered_with'): 'covered_with'}

def apply_reasoning(logical_edges):
    added_new = True
    while added_new:
        added_new = False
        new_edges = set()
        adj = {}
        for u, v, p in logical_edges:
            if u not in adj: adj[u] = []
            adj[u].append((v, p))
            
        for u, v, p in logical_edges:
            if p in SYMMETRIC_PROPERTIES:
                if (v, u, p) not in logical_edges and (v, u, p) not in new_edges: new_edges.add((v, u, p))
            if p in INVERSE_PROPERTIES:
                inv_p = INVERSE_PROPERTIES[p]
                if (v, u, inv_p) not in logical_edges and (v, u, inv_p) not in new_edges: new_edges.add((v, u, inv_p))
            if p in SUB_PROPERTIES:
                sup_p = SUB_PROPERTIES[p]
                if (u, v, sup_p) not in logical_edges and (u, v, sup_p) not in new_edges: new_edges.add((u, v, sup_p))
            if v in adj:
                for w, p2 in adj[v]:
                    if p == p2 and p in TRANSITIVE_PROPERTIES:
                        if (u, w, p) not in logical_edges and (u, w, p) not in new_edges and u != w: new_edges.add((u, w, p))
                    if (p, p2) in PROPERTY_CHAINS:
                        res_p = PROPERTY_CHAINS[(p, p2)]
                        if (u, w, res_p) not in logical_edges and (u, w, res_p) not in new_edges and u != w: new_edges.add((u, w, res_p))
        if new_edges:
            logical_edges.update(new_edges)
            added_new = True
    return logical_edges

def infer_macro_category(name):
    """
    Usa la similarità semantica (cosine similarity) tra la classe rilevata e un insieme di classi note.
    Questo ricalca l'approccio di build_full_set_graphs.py, assicurando coerenza.
    """
    name_lower = name.lower()
    
    if ST_MODEL is None:
        return 'InanimateObject, Item'
        
    cls_emb = ST_MODEL.encode([name_lower], convert_to_tensor=True)
    cos_scores = util.cos_sim(cls_emb, KNOWN_EMBEDDINGS)[0]
    best_idx = torch.argmax(cos_scores).item()
    
    return FLAT_MACRO_MAPPING[best_idx]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM Scene Graph Extractor (Dual Mode)")
    parser.add_argument("--image", required=True, help="Percorso dell'immagine da analizzare")
    parser.add_argument("--mode", choices=["local", "cluster"], default="local", help="Modalità di esecuzione (default: local)")
    parser.add_argument("--model", type=str, default="", help="Nome del modello HF o tag Ollama (es. Qwen/Qwen2-VL-7B-Instruct)")
    parser.add_argument("--output_pt", type=str, default="", help="Percorso dove salvare l'oggetto PyTorch Geometric data (.pt)")
    parser.add_argument("--job_id", type=str, default="inference_job", help="ID del Job associato all'inferenza")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Errore: Immagine non trovata in {args.image}")
        sys.exit(1)
        
    if args.mode == "local":
        model_name = args.model if args.model else "llava:latest"
        graph = extract_graph_ollama(args.image, model_name=model_name)
    else:
        model_id = args.model if args.model else "llava-hf/llava-1.5-13b-hf"
        graph = extract_graph_hf(args.image, model_id=model_id)
        
    if graph:
        print("\n=== SCENE GRAPH ESTRATTO CON SUCCESSO ===")
        print(json.dumps(graph, indent=2))
        
        if args.output_pt:
            # Converte il JSON in PyTorch Geometric data con Arricchimento Ontologico e Reasoning
            node_uri_to_idx = {}
            node_features_text = []
            
            for idx_node, obj in enumerate(graph.get("objects", [])):
                node_id = obj.get("id")
                node_uri_to_idx[node_id] = idx_node
                
                name = obj.get("name", "")
                attrs = obj.get("attributes", [])
                attr_str = ", ".join(attrs) if isinstance(attrs, list) else ""
                macro = infer_macro_category(name)
                
                # Formattazione IDENTICA a quella dei dataset di test e allenamento (build_full_set_graphs.py):
                testo_nomic = f"{name}. Categories: {macro}. Attributes: {attr_str}."
                node_features_text.append(testo_nomic)
                
            num_nodes = len(node_features_text)
            x = torch.randn((num_nodes, 768), dtype=torch.float)
            
            raw_edges = []
            logical_edges = set()
            
            for rel in graph.get("relations", []):
                sogg = rel.get("source") or rel.get("subject")
                ogg = rel.get("target") or rel.get("object")
                proprieta = rel.get("relation") or rel.get("predicate", "")
                
                if sogg in node_uri_to_idx and ogg in node_uri_to_idx:
                    u = node_uri_to_idx[sogg]
                    v = node_uri_to_idx[ogg]
                    raw_edges.append((u, v, proprieta))
                    logical_edges.add((u, v, proprieta))
                    
            # Applicazione del Ragionamento Logico-Ontologico (Sintetico, Inverso, Transitivo)
            expanded_logical_edges = apply_reasoning(logical_edges)
            
            final_source_nodes = []
            final_target_nodes = []
            final_edge_features_text = []
            
            for u, v, p in expanded_logical_edges:
                final_source_nodes.append(u)
                final_target_nodes.append(v)
                final_edge_features_text.append(p)
                
            if len(final_source_nodes) > 0:
                edge_index = torch.tensor([final_source_nodes, final_target_nodes], dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                
            pyg_data = Data(x=x, edge_index=edge_index)
            pyg_data.image_id = args.job_id
            pyg_data.node_text = node_features_text
            pyg_data.edge_text = final_edge_features_text
            
            os.makedirs(os.path.dirname(args.output_pt), exist_ok=True)
            torch.save(pyg_data, args.output_pt)
            print(f"\n[✔] Grafo PyTorch Geometric con arricchimento ontologico salvato in: {args.output_pt}")
        else:
            print("\nNessun output specificato. Il dizionario è pronto per essere convertito.")
    else:
        print("\nFallimento nell'estrazione dello Scene Graph.")
