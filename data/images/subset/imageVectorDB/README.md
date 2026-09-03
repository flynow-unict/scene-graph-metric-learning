# Gallery indicizzata (test gallery)

Le 3.082 immagini dello split `test_gallery`, cioè l'insieme in cui il modello cerca le
scene simili.

Sono servite staticamente dal backend sulla directory montata `/static/vectorDB/` e
alimentano il vector database della demo. A ogni interrogazione il backend confronta
l'embedding a 256 dimensioni della query con quelli di queste immagini, e il frontend
mostra le top-K recuperandole fisicamente da qui.
