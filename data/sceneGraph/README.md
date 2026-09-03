# Scene Graph ed Embedding NOMIC

Contiene la trasformazione dei dati dal formato JSON, estratto dal VLM o da GQA, al formato a grafi atteso dalle reti.

## Struttura

L'albero è diviso su tre livelli: il corpus (`subset` da 15k oppure `fullset`
da 55k), la variante del grafo e lo stadio di elaborazione.

- **`semantic/`**: grafi arricchiti dalle inferenze dell'ontologia OWL su Virtuoso.
- **`baseline/`**: grafi con le sole annotazioni GQA, senza arricchimento. Il
  confronto fra le due varianti isola il contributo del ragionamento ontologico.

Dentro ciascuna variante:

- **`raw/`**: scene graph senza feature di nodo e arco.
- **`embedded/`**: gli stessi grafi tradotti in tensori PyTorch Geometric
  (`Data(x, edge_index, edge_attr, ...)`). È la directory da cui legge la pipeline.
  
### Come sono costruiti i file in `embedded/`
Per ogni split (`train`, `val`, `test_gallery`, `test_queries`):
- `x`: Embedding dei nodi (oggetti). Ogni oggetto è stato dato in pasto all'encoder testuale `nomic-embed-text-v1.5`, generando un vettore a **768 dimensioni**.
- `edge_index`: Matrice delle connessioni (chi è collegato a cosa).
- `edge_attr`: Embedding degli archi (relazioni semantiche). Come per i nodi, l'azione (es. "wearing", "playing with") è embeddata a **768 dimensioni** con Nomic.
