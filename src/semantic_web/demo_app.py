import streamlit as st
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON

# ==========================================
# CONFIGURAZIONE SPARQL
# ==========================================
SPARQL_ENDPOINT = "http://localhost:8893/sparql"
GRAPH_URI = "http://progetto-dl-sw.org/advanced"

# ==========================================
# FUNZIONI DI UTILITÀ
# ==========================================
def run_query(query, inference=False):
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    
    if inference:
        query = 'DEFINE input:inference "advanced_rules"\n' + query
        
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        return results["results"]["bindings"]
    except Exception as e:
        st.error(f"Errore durante l'esecuzione della query: {e}")
        return []

def qname(uri):
    """Semplifica l'URI per la visualizzazione"""
    return uri.split('#')[-1].split('/')[-1]

# ==========================================
# INTERFACCIA STREAMLIT
# ==========================================
st.set_page_config(page_title="GQA Ontology Demo", page_icon="🧠", layout="wide")

st.title("🧠 GQA Knowledge Graph Ontology")
st.markdown("Questa dashboard dimostra le capacità inferenziali dell'ontologia costruita sul dataset GQA.")

tab1, tab2, tab3 = st.tabs(["📊 Statistiche Schema (TBox)", "📈 Statistiche Dati (ABox)", "✨ Demo Ragionatore"])

# ------------------------------------------
# TAB 1: STATISTICHE SCHEMA
# ------------------------------------------
with tab1:
    st.header("Statistiche Ontologiche (TBox)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Statistiche sulle Classi")
        
        query_stats = f"PREFIX owl: <http://www.w3.org/2002/07/owl#> SELECT ?c FROM <{GRAPH_URI}> WHERE {{ ?c a owl:Class . }}"
        res_stats = run_query(query_stats)
        
        if res_stats:
            tot = len(res_stats)
            named = 0
            macro = 0
            anon = 0
            
            for r in res_stats:
                c_val = r['c']['value']
                c_type = r['c']['type']
                
                if c_type == 'bnode':
                    anon += 1
                else:
                    named += 1
                    if c_val.startswith("http://example.invalid/ontologies/ont.owl#"):
                        macro += 1
            
            base = named - macro - 1
            
            st.metric("Classi Totali (Incluse UnionOf)", tot)
            
            st.markdown(f'''
            * **{base}** Classi Base (Oggetti reali del dataset, escludendo *Scene*)
            * **{macro}** Classi Astratte/Info (es. Macroclassi)
            * **{anon}** Classi Anonime (Generate dai blocchi *UnionOf*)
            ''')
            
        st.write("### Gerarchia Macroclassi")
        st.info("Esplora l'albero cliccando sulle macrocategorie per espanderle e visualizzare le sottoclassi.")
        
        with st.spinner("Estrazione alberatura e conteggio istanze in corso..."):
            
            @st.cache_data(show_spinner=False)
            def get_class_counts():
                q = f"""
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX owl: <http://www.w3.org/2002/07/owl#>
                PREFIX ont: <http://progetto-dl-sw.org/ontology#>
                SELECT ?c (COUNT(DISTINCT ?s) AS ?instances) (COUNT(DISTINCT ?img) AS ?images)
                FROM <{GRAPH_URI}>
                WHERE {{
                  ?c a owl:Class .
                  FILTER(isIRI(?c))
                  ?sub rdfs:subClassOf* ?c .
                  ?s a ?sub .
                  OPTIONAL {{ ?img ont:contain ?s }}
                }}
                GROUP BY ?c
                """
                res = run_query(q)
                counts = {}
                if res:
                    for r in res:
                        name = qname(r['c']['value'])
                        counts[name] = {
                            'instances': int(r['instances']['value']),
                            'images': int(r['images']['value'])
                        }
                return counts
                
            class_stats = get_class_counts()
            
            query_tree = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?sub ?super
            FROM <{GRAPH_URI}>
            WHERE {{
              ?sub rdfs:subClassOf ?super .
              FILTER(isIRI(?sub) && isIRI(?super))
              FILTER(STRSTARTS(STR(?super), "http://example.invalid/ontologies/ont.owl#"))
            }}
            """
            res_tree = run_query(query_tree)
            
            if res_tree:
                from collections import defaultdict
                tree = defaultdict(list)
                nodes = set()
                children = set()
                
                for r in res_tree:
                    su = qname(r['super']['value'])
                    sb = qname(r['sub']['value'])
                    tree[su].append(sb)
                    nodes.add(su)
                    nodes.add(sb)
                    children.add(sb)
                
                roots = nodes - children
                
                def render_tree(node, level=0, visited=None):
                    if visited is None:
                        visited = set()
                    
                    if node in visited:
                        return
                    visited.add(node)
                    
                    children_nodes = sorted(tree[node])
                    
                    stats = class_stats.get(node, {'instances': 0, 'images': 0})
                    inst = stats['instances']
                    img = stats['images']
                    
                    if not children_nodes:
                        st.markdown(f"<span style='padding-left: 20px;'>▪️ **{node}** <span style='color: #888; font-size: 0.9em;'>(Istanze: {inst} | Immagini: {img})</span></span>", unsafe_allow_html=True)
                    else:
                        icon = "🟡" if level == 0 else ("🔸" if level == 1 else "🔹")
                        with st.expander(f"{icon} {node} (Istanze: {inst} | Immagini: {img})"):
                            for child in children_nodes:
                                render_tree(child, level + 1, visited.copy())
                                
                for root in sorted(list(roots)):
                    render_tree(root)

    with col2:
        st.subheader("2. Statistiche sulle Proprietà")
        
        query_props = f"""
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        SELECT 
          (COUNT(DISTINCT ?p) AS ?totali)
          (COUNT(DISTINCT IF(EXISTS{{?p a owl:TransitiveProperty}}, ?p, ?unbound)) AS ?transitive)
          (COUNT(DISTINCT IF(EXISTS{{?p a owl:SymmetricProperty}}, ?p, ?unbound)) AS ?symmetric)
          (COUNT(DISTINCT IF(EXISTS{{?p a owl:AsymmetricProperty}}, ?p, ?unbound)) AS ?asymmetric)
          (COUNT(DISTINCT IF(EXISTS{{?p a owl:IrreflexiveProperty}}, ?p, ?unbound)) AS ?irreflexive)
        FROM <{GRAPH_URI}>
        WHERE {{ ?p a owl:ObjectProperty . }}
        """
        res_props = run_query(query_props)
        
        if res_props:
            st.metric("Object Properties Totali", res_props[0]['totali']['value'])
            
            data = {
                "Tipologia": ["Transitive", "Simmetriche", "Asimmetriche", "Irriflessive"],
                "Conteggio": [
                    int(res_props[0]['transitive']['value']),
                    int(res_props[0]['symmetric']['value']),
                    int(res_props[0]['asymmetric']['value']),
                    int(res_props[0]['irreflexive']['value'])
                ]
            }
            st.bar_chart(pd.DataFrame(data).set_index("Tipologia"))
            
        st.write("### Esplora Proprietà (Vista Protégé)")
        if st.button("Carica Dettagli Proprietà"):
            with st.spinner("Caricamento configurazioni..."):
                q_details = f"""
                PREFIX owl: <http://www.w3.org/2002/07/owl#>
                SELECT DISTINCT ?p
                  (EXISTS{{?p a owl:TransitiveProperty}} AS ?isTrans)
                  (EXISTS{{?p a owl:SymmetricProperty}} AS ?isSym)
                  (EXISTS{{?p a owl:AsymmetricProperty}} AS ?isAsym)
                  (EXISTS{{?p a owl:IrreflexiveProperty}} AS ?isIrr)
                FROM <{GRAPH_URI}>
                WHERE {{ ?p a owl:ObjectProperty . }}
                LIMIT 65
                """
                details_res = run_query(q_details)
                
                for r in details_res:
                    p_name = qname(r['p']['value'])
                    with st.expander(f"⚙️ {p_name}"):
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.write("**Caratteristiche (Traits)**")
                            st.checkbox("Transitive", value=(r['isTrans']['value']=='1'), disabled=True, key=f"t_{p_name}")
                            st.checkbox("Symmetric", value=(r['isSym']['value']=='1'), disabled=True, key=f"s_{p_name}")
                            st.checkbox("Asymmetric", value=(r['isAsym']['value']=='1'), disabled=True, key=f"a_{p_name}")
                            st.checkbox("Irreflexive", value=(r['isIrr']['value']=='1'), disabled=True, key=f"i_{p_name}")
                        with c2:
                            st.write("**Domain & Range (Inferred)**")
                            st.info("I Domain e Range specifici utilizzano l'operatore UnionOf (Classe Anonima) per permettere inferenze su più classi di origine, ad es: `Person OR Animals`.")

# ------------------------------------------
# TAB 2: STATISTICHE DATI
# ------------------------------------------
with tab2:
    st.header("Statistiche Dataset (ABox)")
    st.info("Attenzione: le query sui dati possono richiedere qualche secondo a causa della grandezza del grafo.")
    
    colA, colB = st.columns(2)
    
    with colA:
        st.subheader("Popolazione del Grafo")
        if st.button("Calcola Istanze Totali"):
            with st.spinner('Calcolando...'):
                q_instances = f"SELECT (COUNT(DISTINCT ?s) AS ?c) FROM <{GRAPH_URI}> WHERE {{ ?s a ?type . FILTER(STRSTARTS(STR(?s), 'http://progetto-dl-sw.org/objects/')) }}"
                st.metric("Oggetti Istanza Totali", run_query(q_instances)[0]['c']['value'])
                
                q_images = f"SELECT (COUNT(DISTINCT ?s) AS ?c) FROM <{GRAPH_URI}> WHERE {{ ?s a <http://progetto-dl-sw.org/ontology#Scene> }}"
                st.metric("Immagini Totali", run_query(q_images)[0]['c']['value'])

    with colB:
        st.subheader("Complessità delle Immagini")
        st.write("Visualizza le immagini ordinandole per numero di classi (oggetti) e proprietà (relazioni) distinte presenti al loro interno.")
        
        @st.cache_data(show_spinner=False)
        def get_image_complexity_stats():
            q = f"""
            PREFIX ont: <http://progetto-dl-sw.org/ontology#>
            SELECT ?img ?num_classes ?num_props
            FROM <{GRAPH_URI}>
            WHERE {{
              {{
                SELECT ?img (COUNT(DISTINCT ?type) AS ?num_classes)
                WHERE {{
                  ?img a ont:Scene .
                  ?img ont:contain ?s .
                  ?s a ?type .
                  FILTER(STRSTARTS(STR(?type), "http://progetto-dl-sw.org/ontology#"))
                }} GROUP BY ?img
              }}
              {{
                SELECT ?img (COUNT(DISTINCT ?p) AS ?num_props)
                WHERE {{
                  ?img a ont:Scene .
                  ?img ont:contain ?s1 .
                  ?s1 ?p ?o1 .
                  FILTER(STRSTARTS(STR(?p), "http://progetto-dl-sw.org/ontology#"))
                }} GROUP BY ?img
              }}
            }}
            ORDER BY DESC(?num_classes) DESC(?num_props)
            """
            res = run_query(q, inference=False)
            if res:
                return pd.DataFrame([
                    {
                        "ID Immagine": r['img']['value'].split('/')[-1], 
                        "Classi Distinte": int(r['num_classes']['value']),
                        "Proprietà Distinte": int(r['num_props']['value'])
                    } 
                    for r in res
                ])
            return pd.DataFrame()
            
        if st.button("Estrai Complessità Immagini"):
            st.session_state.show_complexity = True
            
        if st.session_state.get("show_complexity", False):
            with st.spinner("Calcolo complessità in corso... (potrebbe richiedere un po' di tempo per tutto il grafo)"):
                df_img = get_image_complexity_stats()
                
            if not df_img.empty:
                st.write("Seleziona una riga per visualizzare l'immagine corrispondente:")
                event = st.dataframe(
                    df_img, 
                    use_container_width=True,
                    selection_mode="single-row",
                    on_select="rerun"
                )
                
                if len(event.selection.rows) > 0:
                    selected_idx = event.selection.rows[0]
                    img_id = df_img.iloc[selected_idx]["ID Immagine"]
                    
                    import os
                    # Risale di due cartelle da src/semantic_web alla radice, poi va in data/images
                    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    img_path = os.path.join(base_dir, "data", "images", f"{img_id}.jpg")
                    
                    if os.path.exists(img_path):
                        st.image(img_path, caption=f"Immagine {img_id}", use_container_width=True)
                    else:
                        st.error(f"Immagine non trovata nel percorso locale: {img_path}")

# ------------------------------------------
# TAB 3: DEMO RAGIONATORE
# ------------------------------------------
with tab3:
    st.header("Dimostrazione Pratica: Il potere dell'Inferenza")
    st.write("In questa sezione confrontiamo i dati grezzi estratti dall'immagine con i dati generati automaticamente dal motore logico di Virtuoso.")
    
    img_id = st.text_input("Inserisci un ID Immagine (Es. 101, 1124, 107970)", value="101")
    
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    img_path = os.path.join(base_dir, "data", "images", f"{img_id}.jpg")
    
    if os.path.exists(img_path):
        st.image(img_path, caption=f"Immagine originale: {img_id}.jpg", width=600)
    else:
        st.warning(f"Anteprima non disponibile: Immagine non trovata nel percorso locale ({img_path})")
    
    if st.button("Mostra Differenza Inferenziale"):
        
        col_raw, col_inf = st.columns(2)
        
        query_str = f"""
        PREFIX ont: <http://progetto-dl-sw.org/ontology#>
        PREFIX img: <http://progetto-dl-sw.org/images/>
        SELECT ?subj_type ?p ?obj_type
        FROM <{GRAPH_URI}>
        WHERE {{
          img:{img_id} ont:contain ?s .
          img:{img_id} ont:contain ?o .
          ?s ?p ?o .
          ?s a ?subj_type .
          ?o a ?obj_type .
          # Filtriamo i metadati inutili per la demo
          FILTER(?p != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)
          FILTER(STRSTARTS(STR(?p), "http://progetto-dl-sw.org/ontology#"))
        }}
        """
        
        with col_raw:
            st.subheader("Dati Grezzi (Senza Inferenza)")
            st.caption("Queste sono le triple esattamente come annotate nel dataset originale.")
            with st.spinner('Estrazione...'):
                raw_res = run_query(query_str, inference=False)
                raw_data = [f"{qname(r['subj_type']['value'])} --[{qname(r['p']['value'])}]--> {qname(r['obj_type']['value'])}" for r in raw_res]
                raw_set = set(raw_data)
                
                if not raw_set:
                    st.warning("Nessuna relazione trovata o immagine inesistente.")
                else:
                    for item in sorted(list(raw_set)):
                        st.write("🔹 " + item)

        with col_inf:
            st.subheader("Dati Ragionati (Con Inferenza)")
            st.caption("Queste includono transitività, property chains e inversi calcolati in tempo reale.")
            with st.spinner('Ragionamento logico in corso...'):
                inf_res = run_query(query_str, inference=True)
                inf_data = [f"{qname(r['subj_type']['value'])} --[{qname(r['p']['value'])}]--> {qname(r['obj_type']['value'])}" for r in inf_res]
                inf_set = set(inf_data)
                
                added_knowledge = inf_set - raw_set
                
                if not inf_set:
                    st.warning("Nessuna relazione trovata.")
                else:
                    st.success(f"Virtuoso ha dedotto {len(added_knowledge)} nuove relazioni logiche!")
                    for item in sorted(list(inf_set)):
                        if item in added_knowledge:
                            st.markdown(f"✅ **<span style='color:green'>{item}</span>** *(Inferito)*", unsafe_allow_html=True)
                        else:
                            st.write("🔹 " + item)
