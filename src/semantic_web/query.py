class GQAQueryFactory:
    """Classe che genera le query SPARQL per il progetto GQA"""

    @staticmethod
    def count_images():
        """Conta quante immagini ci sono nel grafo"""
        return f"""
            PREFIX ont: <http://progetto-dl-sw.org/ontology#>
            PREFIX img: <http://progetto-dl-sw.org/images/>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>

            SELECT (COUNT(DISTINCT ?img) AS ?num_img)
            FROM <http://progetto-dl-sw.org/advanced>
            WHERE{{ 
                ?img a ont:Scene
            }}
        """

    @staticmethod
    def get_scene_ids(limit=None):
        """
        Recupera gli ID delle scene (immagini). 
        Puoi aggiungere un 'LIMIT' per testare con pochi dati.
        """
        limit_clause = f"LIMIT {limit}" if limit else "" # Se passo None, non metto limiti
        
        return f"""
            PREFIX ont: <http://progetto-dl-sw.org/ontology#>
            PREFIX img: <http://progetto-dl-sw.org/images/>
            
            SELECT ?img 
            FROM <http://progetto-dl-sw.org/advanced>
            WHERE {{ 
                ?img a ont:Scene .
            }}
            ORDER BY DESC(STR(?img))
            {limit_clause}
        """

    @staticmethod
    def image_objects_and_attributes(scene_id):
        """
        Recupera tutti gli oggetti e i loro attributi per una scena specifica.
        Santi: questo è il template fondamentale per estrarre i dati grezzi.
        """
        """
        Recupera gli ID delle scene (immagini). 
        Puoi aggiungere un 'LIMIT' per testare con pochi dati.
        """        
        return f"""
            PREFIX ont: <http://progetto-dl-sw.org/ontology#>
            PREFIX img: <http://progetto-dl-sw.org/images/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT ?oggetto ?classeBase 
                (GROUP_CONCAT(DISTINCT ?macroClasse; SEPARATOR=", ") AS ?macroClassi) 
                (GROUP_CONCAT(DISTINCT ?attributo; SEPARATOR=", ") AS ?attributi)
            FROM <http://progetto-dl-sw.org/advanced>
            WHERE {{
            img:{scene_id} ont:contain ?oggetto .
            
            # Trova la classe specifica dell'oggetto
            ?oggetto a ?classeBase .
            FILTER(STRSTARTS(STR(?classeBase), "http://progetto-dl-sw.org/ontology#"))
            
            # Magia del Ragionatore: Trova le super-classi
            OPTIONAL {{
                ?classeBase rdfs:subClassOf* ?macroClasse .
                FILTER(STRSTARTS(STR(?macroClasse), "http://example.invalid/ontologies/ont.owl#"))
                FILTER(?macroClasse != ?classeBase)
            }}
            
            # Trova gli attributi (es. colori)
            OPTIONAL {{ ?oggetto ont:hasAttribute ?attributo }}
            }}
            GROUP BY ?oggetto ?classeBase
        """

    @staticmethod
    def image_relations_classes(scene_id,reasoning=True):
        """
        Recupera tutte le relazioni tra oggetti in una scena specifica.
        Santi: questo è il template fondamentale per estrarre i dati grezzi.
        """
        define_inference="""DEFINE input:inference \"advanced_rules\""""
        if not reasoning:
            define_inference=""
        return f"""
            {define_inference}
            PREFIX ont: <http://progetto-dl-sw.org/ontology#>
            PREFIX img: <http://progetto-dl-sw.org/images/>

            SELECT DISTINCT ?soggetto ?proprieta ?oggettoDestinazione
            FROM <http://progetto-dl-sw.org/advanced>
            WHERE {{
            img:{scene_id} ont:contain ?soggetto, ?oggettoDestinazione .
            ?soggetto ?proprieta ?oggettoDestinazione .
            FILTER(?soggetto != ?oggettoDestinazione)
            FILTER(STRSTARTS(STR(?proprieta), "http://progetto-dl-sw.org/ontology#"))
            FILTER(?proprieta != ont:contain)
            }}
        """