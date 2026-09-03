#!/bin/bash
echo "Attendere l'avvio completo di Virtuoso..."
sleep 5
echo "Avvio caricamento ontologia e dati nel grafo..."
docker exec -i virtuoso_advanced isql 1111 dba santi_semweb_2026 <<EOF
log_enable(2,1);
SPARQL CLEAR GRAPH <http://progetto-dl-sw.org/advanced>;
DELETE FROM DB.DBA.load_list;
ld_dir('/data', '*.ttl', 'http://progetto-dl-sw.org/advanced');
rdf_loader_run();
checkpoint;
rdfs_rule_set('advanced_rules', 'http://progetto-dl-sw.org/advanced');
EOF
echo "Caricamento e inferenza completati con successo!"
echo "Pulizia automatica delle immagini vuote dal database..."
# lanciare dalla radice del progetto:
#   python -m src.utils.delete_empty_images --dry-run
python3 -m src.utils.delete_empty_images
