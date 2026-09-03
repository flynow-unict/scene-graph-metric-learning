# Benchmark completo

Griglia completa di tutte le esecuzioni, nelle stesse colonne riportate dalla
relazione (`relazione/sezioni/04_valutazione_sperimentale.tex`), che resta la
fonte unica: se un numero cambia lì, va aggiornato anche qui.

Le colonne disponibili sono quelle effettivamente riportate in relazione: P@1,
P@5, P@10, R@10, Acc@10, Acc@20. Le altre combinazioni di K non sono state
consolidate e non vengono riportate qui per non lasciare celle non verificate.

Le metriche sono definite in relazione, sezione "Metriche di valutazione del
ranking". A K=1 precisione e hit rate coincidono per costruzione, quindi P@1 vale
anche come Acc@1. Tutti i valori sono percentuali.

Le due varianti di corpus sono **Semantic Web**, con i nodi arricchiti dalle
macro-categorie inferite su Virtuoso, e **GQA nativo**, con le sole etichette
originali del dataset. Le due architetture sono **GINE** (edge-aware, legge
l'etichetta dell'arco) e **SAGE** (sola topologia). In grassetto la riga
`gcn_gine_triplet`, che è il modello di riferimento del lavoro.

---

## 1. Subset 15k, modelli addestrati e valutati su 15k

771 query.

| Configurazione | Architettura | Loss | P@1 | P@5 | P@10 | R@10 | Acc@10 | Acc@20 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Semantic Web | SAGE | NT-Xent | 35.80 | 20.57 | 13.85 | 35.23 | 68.61 | 76.65 |
| Semantic Web | GINE | NT-Xent | 38.39 | 21.74 | 14.51 | 36.38 | 70.04 | 78.34 |
| Semantic Web | SAGE | Triplet | 44.75 | 23.53 | 15.16 | 38.29 | 71.98 | 79.90 |
| Semantic Web | **GINE** | Triplet | **47.34** | **24.85** | **15.71** | **39.85** | **74.45** | **81.58** |
| GQA nativo | SAGE | NT-Xent | 29.70 | 16.78 | 11.35 | 28.88 | 58.11 | 64.46 |
| GQA nativo | GINE | NT-Xent | 31.91 | 17.51 | 11.79 | 30.04 | 60.18 | 67.06 |
| GQA nativo | SAGE | Triplet | 38.26 | 18.86 | 12.39 | 31.63 | 62.78 | 67.70 |
| GQA nativo | **GINE** | Triplet | **40.08** | **19.74** | **12.75** | **32.48** | **63.55** | **68.35** |

## 2. Baseline visive sul subset 15k

Encoder congelati, senza fine-tuning, sullo stesso split.

| Configurazione | Architettura | Loss | P@1 | P@5 | P@10 | R@10 | Acc@10 | Acc@20 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Baseline visiva | ResNet-50 | --- | 23.73 | 10.04 | 6.38 | 16.26 | 36.96 | 44.88 |
| Baseline visiva | CLIP ViT-B/32 | --- | 23.73 | 10.61 | 6.86 | 17.59 | 37.35 | 44.36 |

## 3. Full set 55k, modelli addestrati e valutati su 55k

Oltre 55.000 grafi, 1000 query.

| Configurazione | Architettura | Loss | P@1 | P@5 | P@10 | R@10 | Acc@10 | Acc@20 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Semantic Web | SAGE | NT-Xent | 26.80 | 13.50 | 8.59 | 20.73 | 49.30 | 58.10 |
| Semantic Web | GINE | NT-Xent | 28.70 | 14.56 | 9.39 | 22.52 | 52.40 | 61.50 |
| Semantic Web | SAGE | Triplet | 34.70 | 16.32 | 10.44 | 25.17 | 56.30 | 65.10 |
| Semantic Web | **GINE** | Triplet | **36.90** | **17.28** | **10.95** | **26.41** | **58.80** | **67.20** |
| GQA nativo | SAGE | NT-Xent | 26.30 | 13.10 | 8.46 | 20.39 | 48.50 | 58.10 |
| GQA nativo | GINE | NT-Xent | 28.50 | 14.46 | 9.40 | 22.68 | 51.70 | 61.30 |
| GQA nativo | SAGE | Triplet | 34.90 | 16.58 | 10.46 | 25.25 | 55.30 | 65.20 |
| GQA nativo | **GINE** | Triplet | **37.30** | **17.34** | **10.98** | **26.46** | **59.10** | **67.10** |
| Baseline visiva | ResNet-50 | --- | 16.60 | 6.86 | 4.28 | 10.33 | 29.00 | 37.00 |
| Baseline visiva | CLIP ViT-B/32 | --- | 20.00 | 8.48 | 5.34 | 12.90 | 32.20 | 41.20 |

## 4. Forward cross-evaluation, modelli 55k valutati su 15k

| Configurazione | Architettura | Loss | P@1 | P@5 | P@10 | R@10 | Acc@10 | Acc@20 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Semantic Web | SAGE | NT-Xent | 39.43 | 21.74 | 14.51 | 36.37 | 70.43 | 78.86 |
| Semantic Web | GINE | NT-Xent | 41.50 | 22.67 | 14.98 | 37.86 | 70.82 | 78.99 |
| Semantic Web | SAGE | Triplet | 46.95 | 24.10 | 15.47 | 39.09 | 72.89 | 80.42 |
| Semantic Web | **GINE** | Triplet | **49.68** | **25.08** | **16.02** | **40.50** | **74.71** | **81.71** |
| GQA nativo | SAGE | NT-Xent | 26.85 | 14.58 | 9.77 | 24.87 | 50.06 | 58.11 |
| GQA nativo | GINE | NT-Xent | 27.50 | 15.05 | 10.09 | 25.69 | 52.40 | 60.18 |
| GQA nativo | SAGE | Triplet | 29.70 | 15.64 | 10.57 | 27.04 | 54.73 | 62.78 |
| GQA nativo | **GINE** | Triplet | **31.78** | **16.45** | **10.93** | **27.94** | **56.68** | **63.55** |

## 5. Reverse cross-evaluation, modelli 15k valutati su 55k

Solo configurazione Semantic Web.

| Configurazione | Architettura | Loss | P@1 | P@5 | P@10 | R@10 | Acc@10 | Acc@20 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Semantic Web | SAGE | NT-Xent | 13.40 | 6.86 | 4.46 | 10.76 | 28.70 | 38.20 |
| Semantic Web | GINE | NT-Xent | 15.10 | 7.52 | 4.87 | 11.72 | 30.70 | 40.50 |
| Semantic Web | SAGE | Triplet | 17.50 | 8.78 | 5.63 | 13.57 | 34.70 | 44.80 |
| Semantic Web | **GINE** | Triplet | **18.90** | **9.74** | **6.40** | **15.42** | **38.50** | **49.70** |

## 6. Robustezza a perturbazioni strutturali

Protocollo di self-retrieval su 1.000 scene della gallery, con rimozione casuale
di una quota crescente di nodi. Modello `gcn_gine_triplet` (Semantic Web).

| Nodi rimossi | 0% | 10% | 20% | 30% | 40% |
|---|---:|---:|---:|---:|---:|
| Acc@20 / hit rate | 99.80 | 99.20 | 97.50 | 95.10 | 87.20 |
| P@1 | 99.60 | 95.60 | 89.80 | 79.80 | 64.80 |

Curva in `figures/robustness_test_plot.png`. Fino al 20% di nodi rimossi la
Acc@20 perde poco più di due punti; il calo si concentra su P@1, che scende da
99,60 a 64,80 al 40%.

## 7. Explainability

`GNNExplainer` in modalità regression a livello di grafo, applicato al modello
`gcn_gine_triplet` sulla scena GQA 2363112 (11 nodi, 21 archi). Coefficienti di
salienza dei nodi sopra 0,085:

| Nodo | Salienza |
|---|---:|
| Hair | 0.093 |
| Surfboard | 0.087 |
| Arm | 0.086 |
| Man | 0.085 |

Mappa in `figures/explainability_graph.png`. Gli elementi accessori restano sotto
questa soglia.

Robustezza ed explainability sono state calcolate solo sul subset: ripeterle sul
full set richiederebbe i grafi originali da 55k in locale.

