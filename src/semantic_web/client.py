
from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd

class VirtuosoClient:
    def __init__(self, endpoint="http://localhost:8893/sparql"):
        """
        Inizializza il client per connettersi a Virtuoso.
        Default port per SPARQL: 8890
        """
        self.sparql = SPARQLWrapper(endpoint)
        self.sparql.setReturnFormat(JSON)
        # Se hai impostato una password per l'endpoint SPARQL, scommenta sotto:
        # self.sparql.setCredentials("dba", "santi_semweb_2026")

    def execute_query_json(self, query, timeout=None):
        """
        Esegue una query SPARQL e restituisce i risultati come lista di dizionari.
        """
        self.sparql.setQuery(query)
        # Se c'è un timeout lo impostiamo, altrimenti mettiamo un default alto (10 minuti)
        if timeout is not None:
            self.sparql.setTimeout(timeout)
        else:
            self.sparql.setTimeout(600)

        try:
            results = self.sparql.query().convert()
            
            # Parsing del formato JSON standard di SPARQL
            parsed_results = []
            for result in results["results"]["bindings"]:
                item = {key: result[key]["value"] for key in result.keys()}
                parsed_results.append(item)
            
            return parsed_results
        
        except Exception as e:
            print(f"Errore durante l'esecuzione della query: {e}")
            return None

    def results_to_dataframe(self, results):
        """
        Utility opzionale per convertire i risultati in un DataFrame Pandas,
        molto utile per analisi veloci o per il Deep Learning.
        """
        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results)

# --- ESEMPIO DI UTILIZZO ---

if __name__ == "__main__":
    client = VirtuosoClient()

    # Test Query: Conta le macro-categorie
    #print("Esecuzione query di test...")
    #dati = client.execute_query(test_query)

    #if dati:
        #df = client.results_to_dataframe(dati)
        #print("\nRisultati ottenuti:")
        #print(df)