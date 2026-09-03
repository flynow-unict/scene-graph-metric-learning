#!/bin/bash
mkdir -p data/models/subset/baseline/gcn/

# Definizione dei percorsi
QUERIES="data/sceneGraph/subset/baseline/embedded/test_queries_scene_graphs.pt"
GALLERY="data/sceneGraph/subset/baseline/embedded/test_gallery_scene_graphs.pt"

# Ciclo sui 4 modelli addestrati per il subset baseline
for MODEL in "sage" "sage_triplet" "gine" "gine_triplet"; do
    echo "Estrazione embedding per il modello: $MODEL"
    CKPT="data/models/subset/checkpoints/baseline_gcn_encoder_${MODEL}.pth"
    
    # split test_queries
    OUT_Q="data/models/subset/baseline/gcn/baseline_gcn_test_queries_${MODEL}.pt"
    /home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.extract_gcn_embeddings \
        --split test_queries \
        --graphs "$QUERIES" \
        --ckpt "$CKPT" \
        --out "$OUT_Q"
        
    # split test_gallery
    OUT_G="data/models/subset/baseline/gcn/baseline_gcn_test_gallery_${MODEL}.pt"
    /home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.extract_gcn_embeddings \
        --split test_gallery \
        --graphs "$GALLERY" \
        --ckpt "$CKPT" \
        --out "$OUT_G"
done

echo "Estrazioni subset completate."
