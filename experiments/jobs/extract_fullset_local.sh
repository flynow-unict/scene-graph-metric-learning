#!/bin/bash
mkdir -p data/models/fullset/baseline/gcn/

# Definizione dei percorsi
QUERIES="data/sceneGraph/fullset/baseline/embedded/test_queries_scene_graphs.pt"
GALLERY="data/sceneGraph/fullset/baseline/embedded/test_gallery_scene_graphs.pt"

# Ciclo sui 4 modelli addestrati per il Full Set Baseline
for MODEL in "sage" "sage_triplet" "gine" "gine_triplet"; do
    echo "========================================="
    echo "Estrazione embedding per il modello: $MODEL"
    echo "========================================="
    
    # I checkpoint del Full Set stanno nella sottocartella New/
    CKPT="data/models/fullset/checkpoints/New/baseline_gcn_encoder_${MODEL}.pth"
    
    # split test_queries
    OUT_Q="data/models/fullset/baseline/gcn/baseline_gcn_test_queries_${MODEL}.pt"
    /home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.extract_gcn_embeddings \
        --split test_queries \
        --graphs "$QUERIES" \
        --ckpt "$CKPT" \
        --out "$OUT_Q"
        
    # split test_gallery
    OUT_G="data/models/fullset/baseline/gcn/baseline_gcn_test_gallery_${MODEL}.pt"
    /home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.extract_gcn_embeddings \
        --split test_gallery \
        --graphs "$GALLERY" \
        --ckpt "$CKPT" \
        --out "$OUT_G"
done

echo "Estrazioni Full Set completate."
