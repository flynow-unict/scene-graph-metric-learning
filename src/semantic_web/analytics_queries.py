class GQAQueryFactory:
    """Classe che genera le query SPARQL per il progetto GQA"""

    @staticmethod
    def top_k_popular_classes(k=20):
        """Le K classi più popolose del grafo (escluse macro-categorie)"""
        return f"""
        PREFIX ont: <http://progetto-dl-sw.org/ontology#>
        SELECT ?classe (COUNT(?obj) AS ?numeroIstanze)
        WHERE {{
          ?obj a ?classe .
          FILTER(?classe NOT IN (ont:Item, ont:Nature, ont:Place, ont:Bodypart, ont:Clothing, ont:Person))
        }}
        GROUP BY ?classe
        ORDER BY DESC(?numeroIstanze)
        LIMIT {k}
        """

    @staticmethod
    def top_k_attributes(k=20):
        """I K attributi (colori, materiali, ecc.) più ricorrenti"""
        return f"""
        PREFIX ont: <http://progetto-dl-sw.org/ontology#>
        SELECT ?attributo (COUNT(?attributo) AS ?frequenza)
        WHERE {{
          ?obj ont:hasAttribute ?attributo .
        }}
        GROUP BY ?attributo
        ORDER BY DESC(?frequenza)
        LIMIT {k}
        """

    @staticmethod
    def top_k_properties(k=15):
        """Le K proprietà (relazioni) più ricorrenti (escluse quelle di sistema)"""
        return f"""
        PREFIX ont: <http://progetto-dl-sw.org/ontology#>
        SELECT ?relazione (COUNT(*) AS ?frequenza)
        WHERE {{
          ?s ?relazione ?o .
          FILTER(STRSTARTS(STR(?relazione), "http://progetto-dl-sw.org/ontology#"))
          FILTER(?relazione != ont:hasAttribute)
          FILTER(?relazione != ont:contain)
        }}
        GROUP BY ?relazione
        ORDER BY DESC(?frequenza)
        LIMIT {k}
        """

    @staticmethod
    def top_k_images_by_objects(k=10):
        """Le K immagini che contengono più oggetti"""
        return f"""
        PREFIX ont: <http://progetto-dl-sw.org/ontology#>
        SELECT ?immagine (COUNT(?oggetto) AS ?numeroOggetti)
        WHERE {{
          ?immagine a ont:Scene .
          ?immagine ont:contain ?oggetto .
        }}
        GROUP BY ?immagine
        ORDER BY DESC(?numeroOggetti)
        LIMIT {k}
        """

    @staticmethod
    def food_on_furniture(limit=20):
        """Esempio di ragionamento: Immagini dove il cibo è sopra un mobile"""
        return f"""
        PREFIX ont: <http://progetto-dl-sw.org/ontology#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT DISTINCT ?scena ?cibo ?mobile
        WHERE {{
          ?scena ont:contain ?cibo, ?mobile .
          ?cibo a ?tipoCibo .
          ?tipoCibo rdfs:subClassOf* ont:Food .
          ?mobile a ?tipoMobile .
          ?tipoMobile rdfs:subClassOf* ont:Furniture .
          ?cibo ont:on ?mobile .
        }}
        LIMIT {limit}
        """

    @staticmethod
    def image_objects_and_attributes(image_id):
        """Tutti gli oggetti e attributi di una specifica immagine (ID numerico)"""
        return f"""
        PREFIX ont: <http://progetto-dl-sw.org/ontology#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT ?oggetto ?classe (GROUP_CONCAT(DISTINCT ?attributo; SEPARATOR=", ") AS ?attributi)
        WHERE {{
          <http://progetto-dl-sw.org/images/{image_id}> ont:contain ?oggetto .
          ?oggetto rdf:type ?classe .
          FILTER(STRSTARTS(STR(?classe), "http://progetto-dl-sw.org/ontology#"))
          OPTIONAL {{ ?oggetto ont:hasAttribute ?attributo }}
        }}
        GROUP BY ?oggetto ?classe
        """

    @staticmethod
    def top_k_images_by_relations(k=10):
        """Le K immagini con il grafo più denso (più relazioni tra oggetti)"""
        return f"""
        PREFIX ont: <http://progetto-dl-sw.org/ontology#>
        SELECT ?immagine (COUNT(?rel) AS ?numeroRelazioni)
        WHERE {{
          ?immagine a ont:Scene .
          ?immagine ont:contain ?s, ?o .
          FILTER(?s != ?o)
          ?s ?rel ?o .
          FILTER(STRSTARTS(STR(?rel), "http://progetto-dl-sw.org/ontology#"))
          FILTER(?rel != ont:contain)
        }}
        GROUP BY ?immagine
        ORDER BY DESC(?numeroRelazioni)
        LIMIT {k}
        """

    @staticmethod
    def image_relations_ids(image_id):
        """Tutte le relazioni in un'immagine (ritorna gli ID degli oggetti)"""
        return f"""
        PREFIX ont: <http://progetto-dl-sw.org/ontology#>
        SELECT ?soggetto ?proprieta ?oggettoDestinazione
        WHERE {{
          <http://progetto-dl-sw.org/images/{image_id}> ont:contain ?soggetto, ?oggettoDestinazione .
          ?soggetto ?proprieta ?oggettoDestinazione .
          FILTER(?soggetto != ?oggettoDestinazione)
          FILTER(STRSTARTS(STR(?proprieta), "http://progetto-dl-sw.org/ontology#"))
          FILTER(?proprieta != ont:contain)
        }}
        ORDER BY ?soggetto
        """

    @staticmethod
    def image_relations_classes(image_id):
        """Tutte le relazioni in un'immagine (ritorna i nomi delle classi)"""
        return f"""
        PREFIX ont: <http://progetto-dl-sw.org/ontology#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT ?classeSoggetto ?proprieta ?classeOggetto
        WHERE {{
          <http://progetto-dl-sw.org/images/{image_id}> ont:contain ?soggetto, ?oggettoDestinazione .
          ?soggetto ?proprieta ?oggettoDestinazione .
          ?soggetto rdf:type ?classeSoggetto .
          ?oggettoDestinazione rdf:type ?classeOggetto .
          FILTER(STRSTARTS(STR(?classeSoggetto), "http://progetto-dl-sw.org/ontology#"))
          FILTER(STRSTARTS(STR(?classeOggetto), "http://progetto-dl-sw.org/ontology#"))
          FILTER(?soggetto != ?oggettoDestinazione)
          FILTER(STRSTARTS(STR(?proprieta), "http://progetto-dl-sw.org/ontology#"))
          FILTER(?proprieta != ont:contain)
        }}
        ORDER BY ?classeSoggetto
        """