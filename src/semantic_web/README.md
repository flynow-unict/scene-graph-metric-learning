# Progetto Semantic Web - Dataset GQA (Subsample)

Questo pacchetto contiene tutto il necessario per avviare l'infrastruttura del Knowledge Graph basato sul dataset GQA e testare le funzionalità di inferenza tramite la dashboard interattiva.

## Note di Compatibilità
L'intero ecosistema è stato sviluppato e testato in ambiente **Linux (Debian)**, ma l'utilizzo di container Docker e script Python garantisce la piena compatibilità **cross-platform**. Il progetto funzionerà in modo identico anche su **Windows** o **macOS**.

---

## Prerequisiti
Per eseguire il progetto su qualsiasi dispositivo, è necessario avere installati:
1. **Docker** e **Docker Compose** (per l'avvio del database Virtuoso).
2. **Python 3.9+** (per l'esecuzione della dashboard Streamlit).

---

## 1. Avvio del Database (Virtuoso)
Il triple store Virtuoso è configurato per essere avviato tramite Docker, in modo da garantire la massima portabilità senza dover installare server locali.

1. Aprire un terminale all'interno di questa cartella (`consegna`).
2. Avviare il container in background:
   ```bash
   docker compose up
   ```
3. Attendere qualche secondo per consentire a Docker di avviare il database.

---

## 2. Caricamento dell'Ontologia e dei Dati
I file RDF (schema e dati) si trovano nella cartella `virtuoso/share`. Per caricarli automaticamente nel grafo di Virtuoso e impostare le regole logiche di inferenza (RDFS / OWL), è stato predisposto uno script di automazione.

1. Sempre dal terminale, eseguire lo script di inizializzazione:
   ```bash
   ./init_virtuoso.sh
   ```
   *Nota per gli utenti Windows: se l'esecuzione dello script Bash dovesse dare problemi (es. WSL non configurato), è possibile inserire i dati eseguendo i comandi SQL manualmente.*

Se lo script termina con successo, il Knowledge Graph sarà pronto e le regole di ragionamento avanzato ("advanced_rules") saranno attive all'indirizzo `http://localhost:8893/sparql`.

---

## 3. Configurazione Ambiente Python e Librerie
Per avviare la demo visiva basata su Streamlit, è necessario installare alcune librerie specifiche.
**È fortemente consigliato** l'utilizzo di un gestore di pacchetti per creare un ambiente virtuale isolato, come **Conda** o **venv**.

### Creazione Ambiente (Scegliere un'opzione)

**Opzione A (Conda):**
```bash
conda create -n sw_project python=3.10 -y
conda activate sw_project
```

**Opzione B (Venv standard):**
```bash
python -m venv venv
source venv/bin/activate  # (Su Windows: venv\Scripts\activate)
```

### Installazione Requisiti
Una volta attivato l'ambiente, installare le dipendenze essenziali:
```bash
pip install streamlit pandas sparqlwrapper
```

---

## 4. Avvio della Dashboard (Streamlit)
I percorsi per il caricamento delle 16.000 immagini del subsample sono stati configurati in modo dinamico e relativo. **Non c'è bisogno di modificare alcun file o percorso nel codice**.

Per lanciare la demo interattiva, dal terminale con l'ambiente virtuale attivo, eseguire:
```bash
streamlit run demo_app.py
```
Il browser si aprirà automaticamente (oppure visitate l'indirizzo `http://localhost:8501`). 

### Funzionalità della Dashboard:
- **Tab 1:** Consultare le statistiche aggiornate della TBox, esplorare gerarchie e visualizzare i tratti semantici in stile Protégé.
- **Tab 2:** Visualizzare le classifiche di complessità delle immagini (ABox) e caricare dinamicamente le relative foto.
- **Tab 3:** Provare il "Ragionatore Logico" per vedere l'inferenza in tempo reale delle proprietà (transitività, inversioni, ecc.) applicata alle singole immagini.

---

## Informazioni sul Dataset
Il dataset incluso in questo pacchetto (`dati.ttl`) è un subsample rigorosamente bilanciato derivato dal dataset originale GQA della Stanford University. Comprende:
- **300** Classi base (Oggetti) + 20 Macro-categorie Gerarchiche
- **16.000** Immagini descritte
- Regole inferenziali complesse (Property Chains, Transitività, UnionOf, InverseOf, ecc.).
