"""Calcolo della similarità fra scene per identificare le coppie dello "stesso scenario".

Questa logica è utilizzata per due scopi che devono condividere la stessa metrica:
  - training contrastivo: selezione dei positivi (scene simili all'ancora);
  - valutazione retrieval: ground-truth di rilevanza per calcolare Accuracy/Precision/Recall.

Poiché il dataset GQA non ha etichette esplicite per lo scenario, la similarità viene 
ricavata dal contenuto dei grafi. I grafi originali non vengono modificati in questa fase:
questa funzione serve solo come criterio di valutazione per accoppiare le scene, 
mentre il modello (GCN) continuerà a ricevere in input nodi, archi e feature originali.

Tre modalità supportate (parametro `mode`):
  - "objects": Indice di Jaccard sull'insieme degli oggetti (le ~300 classi base).
               Fornisce un risultato denso ma ignora la struttura del grafo.
  - "triples": Indice di Jaccard sulle triple semantiche (vedi sotto). 
               Considera la struttura ma produce risultati molto sparsi.
  - "mixed"  : Media pesata delle due modalità precedenti (default). 
               Unisce i vantaggi di entrambe: risultato denso e sensibile alla struttura.

Una "tripla" è una relazione del tipo: (classe_sorgente, relazione, classe_destinazione).
Consideriamo solo le triple "semantiche": scartiamo le relazioni puramente spaziali
(come left/right of, near, above...) perché dipendono dall'inquadratura dell'immagine 
e non aiutano a definire lo scenario complessivo.

Perché "mixed" è la scelta migliore (statistiche sul val set, 1541 grafi):
  metric        rnd_mean  sat%=1  cov>=1  ties@top  struct_r
  objects        0.045     0.0%    92%      1.3       0.30
  macro(19)      0.385     0.3%    86%      5.5       0.04   <- troppo generica
  triples-sem    0.001     0.0%    55%    694.3       1.00   <- troppo sparsa
  feat-cosine    0.000     0.0%   100%      1.0       0.00   <- non strutturale
  mixed          0.023     0.0%    92%      1.2       0.30   <- densa, strutturale, ordinabile
(cov>=1 = query con almeno un positivo; ties@top = parimerito in cima; 
 struct_r = correlazione con la struttura). L'opzione mixed è l'unica che tiene
conto in modo bilanciato degli archi, incentivando il modello a utilizzarli.
"""
from __future__ import annotations

import math
from collections import Counter

# Relazioni puramente spaziali: dipendono dall'inquadratura, non dallo scenario.
SPATIAL_RELATIONS = {
    "to_the_left_of", "to_the_right_of", "in_front_of", "behind", "near",
    "next_to", "above", "below", "on_top_of", "under", "beside", "by", "at",
}

# Oggetti "generici" che non caratterizzano lo scenario (distinzione stuff-vs-things):
# Sfondi comuni (sky, ground, wall) e concetti generici legati a persone (man, head, hand).
# Essendo onnipresenti, potrebbero generare falsi positivi di similarità (es. due scene 
# distinte accomunate solo dalla presenza di cielo e di una persona).
# NB: Non includiamo sfondi fortemente caratterizzanti (snow, water, sand, grass, road). 
# Questa raccolta agisce come una stop-list nel Natural Language Processing.
STOP_OBJECTS = {
    # sfondo / stuff generico
    "sky", "ground", "tree", "trees", "wall", "air", "floor", "ceiling",
    "background", "shadow", "reflection", "light", "lights",
    # capi d'abbigliamento generici
    "clothes", "shirt", "pants", "jacket",
    # persone generiche
    "man", "woman", "person", "people", "guy", "girl", "boy", "kid", "child", "lady",
    # parti del corpo
    "hair", "head", "hand", "hands", "face", "eye", "eyes", "leg", "legs",
    "arm", "arms", "nose", "mouth", "ear", "foot", "feet", "shoe", "shoes",
}


def _base_class(node_text: str) -> str:
    """Da 'Sky. Categories: ... Attributes: ...' estrae la classe-oggetto 'sky'."""
    return node_text.split(".")[0].strip().lower()


def object_set(graph) -> set:
    """Insieme delle classi-oggetto presenti nella scena (le ~300 classi base)."""
    return {_base_class(t) for t in graph.node_text}


def semantic_triple_set(graph) -> set:
    """Insieme delle triple (classe_sorgente, relazione, classe_dest), spaziali escluse."""
    cls = [_base_class(t) for t in graph.node_text]
    edge_index = graph.edge_index
    triples = set()
    for e in range(edge_index.size(1)):
        s = int(edge_index[0, e])
        o = int(edge_index[1, e])
        rel = graph.edge_text[e] if e < len(graph.edge_text) else "?"
        if rel in SPATIAL_RELATIONS:
            continue
        if s < len(cls) and o < len(cls):
            triples.add((cls[s], rel, cls[o]))
    return triples


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_idf(signatures, stop_objects=STOP_OBJECTS, stop_damp: float = 0.15):
    """Assegna un peso (IDF) a ogni oggetto e tripla, riducendo l'impatto dei token generici.

    Il valore IDF è alto per i token rari (es. horse, snow) che aiutano a discriminare,
    e basso per quelli comuni (es. man, sky). Tuttavia, l'IDF da solo non distingue
    un elemento "raro ma significativo" (kite) da uno "raro ma generico" (clothes, guy).
    Per questo motivo, il peso dei token elencati in `stop_objects` (sfondi, persone, parti 
    del corpo) viene ulteriormente ridotto moltiplicandolo per `stop_damp` (0.15). 
    
    Una tripla viene penalizzata solo se ENTRAMBI i suoi estremi sono generici, in modo che
    le azioni significative (es. (man, riding, horse)) mantengano un peso alto, mentre 
    le relazioni marginali (es. (man, wearing, clothes)) contino di meno. 
    In questo modo non rimuoviamo nessun dato dal grafo (evitando che scene con pochi oggetti 
    perdano troppe informazioni), ma bilanciamo l'importanza dei vari elementi. 
    I pesi vanno calcolati sull'intero corpus di riferimento.
    """
    n = len(signatures)
    df_o, df_t = Counter(), Counter()
    for obj, tri in signatures:
        for x in obj:
            df_o[x] += 1
        for x in tri:
            df_t[x] += 1
    idf_o = {k: math.log((n + 1) / (v + 1)) + 1 for k, v in df_o.items()}
    idf_t = {k: math.log((n + 1) / (v + 1)) + 1 for k, v in df_t.items()}
    if stop_objects:
        idf_o = {k: (v * stop_damp if k in stop_objects else v) for k, v in idf_o.items()}
        idf_t = {k: (v * stop_damp if (k[0] in stop_objects and k[2] in stop_objects) else v)
                 for k, v in idf_t.items()}
    return idf_o, idf_t


def weighted_jaccard(a: set, b: set, weights: dict) -> float:
    """Jaccard pesata: somma dei pesi dell'intersezione / somma dei pesi dell'unione."""
    union = a | b
    if not union:
        return 0.0
    inter = sum(weights.get(x, 1.0) for x in (a & b))
    denom = sum(weights.get(x, 1.0) for x in union)
    return inter / denom if denom else 0.0


def graph_signature(graph):
    """Precalcola la firma (object_set, triple_set) usata per il confronto.

    La firma è fedele al contenuto originale: non viene rimosso alcun elemento. 
    L'impatto dei token generici viene invece gestito tramite una pesatura morbida 
    in `compute_idf`. Dai test effettuati, l'eliminazione diretta dei nodi riduceva 
    eccessivamente il contenuto del grafo compromettendo i match validi."""
    return object_set(graph), semantic_triple_set(graph)


def similarity_from_signatures(sig_a, sig_b, mode: str = "mixed", alpha: float = 0.5,
                               weights=None, min_triples: int = 0) -> float:
    """Similarita' a partire dalle firme precalcolate. alpha = peso di objects in 'mixed'.

    `weights` = (idf_oggetti, idf_triple) da compute_idf: se dato, usa la Jaccard pesata
    (i token rari contano di piu'); se None, Jaccard classica non pesata.

    `min_triples`: in 'mixed', se uno dei due grafi ha meno di min_triples triple la
    meta' strutturale viene ignorata (si usa solo objects). Con 1-2 triple il denominatore
    della Jaccard e' cosi' piccolo che UNA coincidenza rara ma generica (es. man-wearing-cap)
    domina il punteggio; sotto questa soglia la stima strutturale non e' affidabile.
    """
    obj_a, tri_a = sig_a
    obj_b, tri_b = sig_b
    if mode == "mixed" and min_triples and (len(tri_a) < min_triples or len(tri_b) < min_triples):
        mode = "objects"
    if weights is None:
        j_obj = lambda: jaccard(obj_a, obj_b)
        j_tri = lambda: jaccard(tri_a, tri_b)
    else:
        w_obj, w_tri = weights
        j_obj = lambda: weighted_jaccard(obj_a, obj_b, w_obj)
        j_tri = lambda: weighted_jaccard(tri_a, tri_b, w_tri)
    if mode == "objects":
        return j_obj()
    if mode == "triples":
        return j_tri()
    if mode == "mixed":
        return alpha * j_obj() + (1.0 - alpha) * j_tri()
    raise ValueError(f"mode sconosciuto: {mode!r} (usa 'objects', 'triples' o 'mixed')")


def scene_similarity(graph_a, graph_b, mode: str = "mixed", alpha: float = 0.5,
                     weights=None) -> float:
    """Similarita' fra due scene in [0, 1]. mode in {'objects','triples','mixed'}.

    Nota: per la Jaccard pesata (consigliata) i pesi IDF vanno calcolati sull'intero
    corpus, quindi qui `weights` va passato dall'esterno (una coppia isolata non ha IDF).
    """
    return similarity_from_signatures(
        graph_signature(graph_a), graph_signature(graph_b),
        mode=mode, alpha=alpha, weights=weights,
    )


def compute_scene_embeddings(graphs, weights=None):
    """Calcola un embedding semantico per ogni scena facendo la media (normalizzata)
    degli embedding NOMIC dei nodi (graph.x). 
    
    Questo serve come controllo aggiuntivo (veto semantico): poiché la metrica Jaccard 
    cerca solo corrispondenze esatte, rischia di dare un punteggio alto a scene che 
    condividono pochi oggetti casuali ma parlano di cose diverse. La distanza coseno 
    degli embedding, essendo più flessibile sui sinonimi, aiuta a filtrare questi errori.

    Se viene fornito il parametro `weights` (idf_oggetti da compute_idf), viene
    eseguita una media pesata. Senza pesi, l'embedding finale di una scena verrebbe
    influenzato troppo da oggetti molto comuni (come cielo, mani, persone), nascondendo
    i soggetti più distintivi (es. aquilone, cavallo). Usando i pesi, invece, l'embedding 
    dà maggiore priorità agli elementi che caratterizzano davvero la scena.

    Gli embedding vengono centrati sulla media del corpus prima di essere normalizzati.
    Questo passaggio è necessario poiché i vettori NOMIC grezzi tendono a concentrarsi in 
    una regione ristretta dello spazio (anisotropia), rendendo il calcolo del coseno 
    meno efficace senza un riposizionamento preliminare."""
    import torch
    if weights is None:
        embs = torch.stack([g.x.mean(dim=0) for g in graphs])
    else:
        pooled = []
        for g in graphs:
            w = torch.tensor([weights.get(_base_class(t), 1.0) for t in g.node_text],
                             dtype=g.x.dtype)
            pooled.append((g.x * w.unsqueeze(1)).sum(dim=0) / w.sum())
        embs = torch.stack(pooled)
    embs = embs - embs.mean(dim=0)
    return torch.nn.functional.normalize(embs, dim=1)


def build_inverted_index(signatures):
    """object -> lista di indici-grafo che la contengono. Per non confrontare tutto con tutto."""
    from collections import defaultdict
    inv = defaultdict(list)
    for i, (obj, _tri) in enumerate(signatures):
        for o in obj:
            inv[o].append(i)
    return inv


def mine_positives(graphs, mode: str = "mixed", alpha: float = 0.5, min_sim: float = 0.15,
                   signatures=None, n_rare: int = 6, weighted: bool = True, weights=None,
                   k: int = 5, embeddings=None, sem_veto: float = 0.0,
                   sim_strong: float = 0.30, min_objects: int = 3, min_triples: int = 0):
    """Per ogni grafo trova i k positivi migliori (le scene più simili, esclusa se stessa).

    Restituisce una lista di lunghezza len(graphs): per ogni ancora, una lista di 
    tuple (pos_idx, score) che superano `min_sim`, ordinate per punteggio decrescente, 
    fino a un massimo di `k` elementi. Se la lista è vuota, significa che non ci sono 
    positivi affidabili.
    Nota: restituiamo al massimo i top-k elementi CHE SUPERANO LA SOGLIA. Se un'ancora
    ha solo 2 scene realmente simili nel corpus, ne restituiremo 2, evitando di aggiungere
    dati rumorosi.

    Se `weighted=True` (default), utilizza la metrica Jaccard pesata tramite IDF: in 
    questo modo, le corrispondenze su oggetti generici non generano falsi positivi.

    I parametri `embeddings` e `sem_veto` attivano il controllo semantico aggiuntivo:
    se un candidato ha una similarità Jaccard moderata (sotto `sim_strong`), viene
    incluso nei top-k solo se la distanza coseno rispetto all'ancora è maggiore o 
    uguale a `sem_veto`. Sopra `sim_strong`, la corrispondenza esatta è considerata 
    sufficiente. Questo aiuta a filtrare le false corrispondenze dovute alla 
    coincidenza di un singolo oggetto.

    Ottimizzazione: per evitare un confronto computazionalmente oneroso (O(N^2)), 
    tramite un indice invertito vengono analizzati solo i grafi che condividono con 
    l'ancora almeno uno degli oggetti più rari (`n_rare`). Solo questo sottoinsieme 
    di candidati viene poi valutato in modo completo.
    """
    if signatures is None:
        signatures = [graph_signature(g) for g in graphs]
    if weighted and weights is None:
        weights = compute_idf(signatures)
    elif not weighted:
        weights = None
    inv = build_inverted_index(signatures)
    results = []
    for i, (obj_i, _tri_i) in enumerate(signatures):
        # Ancora con carenza di informazioni: grafi con 1-2 oggetti non sono sufficienti
        # a identificare uno scenario univoco (es. una scena "flower+leaves" potrebbe 
        # corrispondere erroneamente a un "ombrellone a fiori"). I test visivi tramite 
        # CLIP indicano una precisione del 30% per ancore con <=2 oggetti, contro il 
        # 54-62% dal terzo oggetto in su. Si preferisce scartarle per evitare falsi match.
        if len(obj_i) < min_objects:
            results.append([])
            continue
        rare = sorted(obj_i, key=lambda o: len(inv[o]))[:n_rare]  # oggetti più discriminativi
        cand = set()
        for o in rare:
            cand.update(inv[o])
        cand.discard(i)
        
        # OTTIMIZZAZIONE: limitiamo i candidati per evitare esplosione O(N^2) su 55k grafi
        import random
        if len(cand) > 1000:
            cand = set(random.sample(list(cand), 1000))
            
        scored = []
        for j in cand:
            s = similarity_from_signatures(signatures[i], signatures[j], mode=mode,
                                           alpha=alpha, weights=weights,
                                           min_triples=min_triples)
            if s <= min_sim:
                continue
            if (s < sim_strong and embeddings is not None
                    and float(embeddings[i] @ embeddings[j]) < sem_veto):
                continue
            scored.append((j, s))
        scored.sort(key=lambda t: t[1], reverse=True)
        results.append(scored[:k])
    return results


def topk_similar(query_idx, graphs, k: int = 5, mode: str = "mixed", alpha: float = 0.5,
                 signatures=None, weighted: bool = True, weights=None,
                 embeddings=None, sem_veto: float = 0.0, sim_strong: float = 0.30,
                 min_triples: int = 0):
    """Indici + punteggi dei k grafi piu' simili all'ancora (esclusa se stessa).

    Ritorna lista di (idx, score) ordinata per score decrescente. `signatures` puo'
    essere una lista di firme precalcolate (da graph_signature) per andare piu' veloce.
    `weighted=True` (default) usa la Jaccard pesata IDF sul corpus dato.
    `embeddings` + `sem_veto`: veto semantico come in mine_positives (il candidato deve
    avere coseno NOMIC >= sem_veto con l'ancora per poter entrare nei top-k).
    """
    if signatures is None:
        signatures = [graph_signature(g) for g in graphs]
    if weighted and weights is None:
        weights = compute_idf(signatures)
    elif not weighted:
        weights = None
    q = signatures[query_idx]
    scored = []
    for j in range(len(graphs)):
        if j == query_idx:
            continue
        s = similarity_from_signatures(q, signatures[j], mode=mode, alpha=alpha,
                                       weights=weights, min_triples=min_triples)
        if (s < sim_strong and embeddings is not None
                and float(embeddings[query_idx] @ embeddings[j]) < sem_veto):
            continue
        scored.append((j, s))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:k]
