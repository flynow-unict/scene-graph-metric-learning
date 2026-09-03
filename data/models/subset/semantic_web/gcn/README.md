# Embedding del graph encoder

Embedding a 256 dimensioni estratti dai modelli addestrati sui grafi arricchiti, sugli
split di valutazione del corpus ridotto.

## File

Otto file, uno per ogni combinazione di architettura, loss e split.

| File | Architettura | Loss | Split | Righe | Dim |
|---|---|---|---|---:|---:|
| `gcn_gine_test_gallery.pt` | GINEConv | NT-Xent | gallery | 3082 | 256 |
| `gcn_gine_test_queries.pt` | GINEConv | NT-Xent | queries | 771 | 256 |
| `gcn_gine_triplet_test_gallery.pt` | GINEConv | Triplet | gallery | 3082 | 256 |
| `gcn_gine_triplet_test_queries.pt` | GINEConv | Triplet | queries | 771 | 256 |
| `gcn_sage_test_gallery.pt` | SAGEConv | NT-Xent | gallery | 3082 | 256 |
| `gcn_sage_test_queries.pt` | SAGEConv | NT-Xent | queries | 771 | 256 |
| `gcn_sage_triplet_test_gallery.pt` | SAGEConv | Triplet | gallery | 3082 | 256 |
| `gcn_sage_triplet_test_queries.pt` | SAGEConv | Triplet | queries | 771 | 256 |

GINEConv legge l'etichetta dell'arco, SAGEConv la ignora e lavora sulla sola topologia.

I file `_test_gallery` sono l'insieme in cui cercare, i `_test_queries` le
interrogazioni: il ranking si ottiene dalla cosine similarity fra i due. La stessa
dimensionalità, 256 valori normalizzati in norma L2, è quella prodotta dall'encoder in
inferenza nel worker cloud, per cui questi file servono anche a validare l'output del
servizio online prima che sia in rete.

Il formato del dizionario è lo stesso delle baseline visive, con `"encoder": "gcn"`.

## Ground truth di rilevanza

La ground truth query→gallery sta in `../relevance_test.pt`, generata da
`src/evaluation/build_relevance.py`. La similarità non è una classe discreta ma un valore
continuo, ottenuto dalla Jaccard pesata con IDF sugli oggetti e sulle triple e filtrato
dal veto semantico sul coseno NOMIC.

- `relevance` — matrice `[771, 3082]` di score continui.
- `binary` — la stessa matrice discretizzata a soglia.
- `meta` — i parametri usati per generarla.
