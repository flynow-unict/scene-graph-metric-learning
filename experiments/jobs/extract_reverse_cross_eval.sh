#!/bin/bash
set -e

# Definisci le directory
CKPT_DIR="checkpoints/subset/checkpoints"
OUT_DIR="data/cross_evaluation_reverse"
mkdir -p "$OUT_DIR"

# Path ai grafi Full Set (Semantic Web)
REL_GALLERY="data/sceneGraph/embedded/full_set/test_gallery_scene_graphs.pt"
REL_QUERIES="data/sceneGraph/embedded/full_set/test_queries_scene_graphs.pt"

echo "=== Estrazione Modelli Sub-Dataset (15k) sui Dati Full Set (55k) ==="

echo "-> GINE"
/home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.extract_gcn_embeddings --graphs "$REL_GALLERY" --ckpt "$CKPT_DIR/gcn_encoder_gine.pth" --out "$OUT_DIR/reverse_test_gallery_gine.pt"
/home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.extract_gcn_embeddings --graphs "$REL_QUERIES" --ckpt "$CKPT_DIR/gcn_encoder_gine.pth" --out "$OUT_DIR/reverse_test_queries_gine.pt"

echo "-> SAGE"
/home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.extract_gcn_embeddings --graphs "$REL_GALLERY" --ckpt "$CKPT_DIR/gcn_encoder_sage.pth" --out "$OUT_DIR/reverse_test_gallery_sage.pt"
/home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.extract_gcn_embeddings --graphs "$REL_QUERIES" --ckpt "$CKPT_DIR/gcn_encoder_sage.pth" --out "$OUT_DIR/reverse_test_queries_sage.pt"

echo "-> GINE TRIPLET"
/home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.extract_gcn_embeddings --graphs "$REL_GALLERY" --ckpt "$CKPT_DIR/gcn_encoder_gine_triplet.pth" --out "$OUT_DIR/reverse_test_gallery_gine_triplet.pt"
/home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.extract_gcn_embeddings --graphs "$REL_QUERIES" --ckpt "$CKPT_DIR/gcn_encoder_gine_triplet.pth" --out "$OUT_DIR/reverse_test_queries_gine_triplet.pt"

echo "-> SAGE TRIPLET"
/home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.extract_gcn_embeddings --graphs "$REL_GALLERY" --ckpt "$CKPT_DIR/gcn_encoder_sage_triplet.pth" --out "$OUT_DIR/reverse_test_gallery_sage_triplet.pt"
/home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.extract_gcn_embeddings --graphs "$REL_QUERIES" --ckpt "$CKPT_DIR/gcn_encoder_sage_triplet.pth" --out "$OUT_DIR/reverse_test_queries_sage_triplet.pt"

echo "Estrazione completata in $OUT_DIR"
