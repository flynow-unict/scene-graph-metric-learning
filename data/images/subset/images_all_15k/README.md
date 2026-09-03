# Immagini GQA (corpus ridotto)

Immagini RAW in formato `.jpg` del dataset GQA originale, le stesse 15k scene su cui
girano gli esperimenti del corpus ridotto.

## Uso nella pipeline

- **Baseline visive** — sono l'input degli estrattori ResNet50 e CLIP, che leggono
  direttamente dai pixel senza passare dallo scene graph.
- **Frontend** — sono le immagini mostrate all'utente nell'interfaccia cloud.

## Formato dei file

Ogni file prende il nome dal proprio `image_id` originale, ad esempio `1159275.jpg`.
L'`image_id` è la chiave con cui gli embedding, testuali o visivi, vengono ricollegati
all'immagine corretta in tutte le altre cartelle e in tutti i file `.pt`.
