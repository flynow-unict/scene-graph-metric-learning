#!/bin/bash
set -e

# Definisci le directory
CKPT_DIR="data/models/fullset/checkpoints"
OUT_DIR="data/cross_evaluation"
mkdir -p "$OUT_DIR"

# Path ai grafi originali (15k)
NOREL_GALLERY="data/sceneGraph/embedded/NoRel/test_gallery_embedded.pt"
NOREL_QUERIES="data/sceneGraph/embedded/NoRel/test_queries_embedded.pt"
REL_GALLERY="data/sceneGraph/embedded/Rel/test_gallery_scene_graphs.pt"
REL_QUERIES="data/sceneGraph/embedded/Rel/test_queries_scene_graphs.pt"

echo "=== Estrazione Modelli Full Set sui Dati Sub-Dataset (15k) ==="

# 1. Baseline (NoRel)
echo "-> GINE Baseline"
/home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.extract_gcn_embeddings --graphs "$NOREL_GALLERY" --ckpt "$CKPT_DIR/baseline_gcn_encoder_gine.pth" --out "$OUT_DIR/baseline_cross_test_gallery_gine.pt"
/home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.extract_gcn_embeddings --graphs "$NOREL_QUERIES" --ckpt "$CKPT_DIR/baseline_gcn_encoder_gine.pth" --out "$OUT_DIR/baseline_cross_test_queries_gine.pt"

echo "-> SAGE Baseline"
/home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.extract_gcn_embeddings --graphs "$NOREL_GALLERY" --ckpt "$CKPT_DIR/baseline_gcn_encoder_sage.pth" --out "$OUT_DIR/baseline_cross_test_gallery_sage.pt"
/home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.extract_gcn_embeddings --graphs "$NOREL_QUERIES" --ckpt "$CKPT_DIR/baseline_gcn_encoder_sage.pth" --out "$OUT_DIR/baseline_cross_test_queries_sage.pt"

echo "-> GINE TRIPLET Baseline"
/home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.extract_gcn_embeddings --graphs "$NOREL_GALLERY" --ckpt "$CKPT_DIR/baseline_gcn_encoder_gine_triplet.pth" --out "$OUT_DIR/baseline_cross_test_gallery_gine_triplet.pt"
/home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.extract_gcn_embeddings --graphs "$NOREL_QUERIES" --ckpt "$CKPT_DIR/baseline_gcn_encoder_gine_triplet.pth" --out "$OUT_DIR/baseline_cross_test_queries_gine_triplet.pt"

echo "-> SAGE TRIPLET Baseline"
/home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.extract_gcn_embeddings --graphs "$NOREL_GALLERY" --ckpt "$CKPT_DIR/baseline_gcn_encoder_sage_triplet.pth" --out "$OUT_DIR/baseline_cross_test_gallery_sage_triplet.pt"
/home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.extract_gcn_embeddings --graphs "$NOREL_QUERIES" --ckpt "$CKPT_DIR/baseline_gcn_encoder_sage_triplet.pth" --out "$OUT_DIR/baseline_cross_test_queries_sage_triplet.pt"

# 2. Semantic Web (Rel)
echo "-> GINE Semantic Web"
/home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.extract_gcn_embeddings --graphs "$REL_GALLERY" --ckpt "$CKPT_DIR/gcn_encoder_gine.pth" --out "$OUT_DIR/gcn_cross_test_gallery_gine.pt"
/home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.extract_gcn_embeddings --graphs "$REL_QUERIES" --ckpt "$CKPT_DIR/gcn_encoder_gine.pth" --out "$OUT_DIR/gcn_cross_test_queries_gine.pt"

echo "-> SAGE Semantic Web"
/home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.extract_gcn_embeddings --graphs "$REL_GALLERY" --ckpt "$CKPT_DIR/gcn_encoder_sage.pth" --out "$OUT_DIR/gcn_cross_test_gallery_sage.pt"
/home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.extract_gcn_embeddings --graphs "$REL_QUERIES" --ckpt "$CKPT_DIR/gcn_encoder_sage.pth" --out "$OUT_DIR/gcn_cross_test_queries_sage.pt"

echo "-> GINE TRIPLET Semantic Web"
/home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.extract_gcn_embeddings --graphs "$REL_GALLERY" --ckpt "$CKPT_DIR/gcn_encoder_gine_triplet.pth" --out "$OUT_DIR/gcn_cross_test_gallery_gine_triplet.pt"
/home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.extract_gcn_embeddings --graphs "$REL_QUERIES" --ckpt "$CKPT_DIR/gcn_encoder_gine_triplet.pth" --out "$OUT_DIR/gcn_cross_test_queries_gine_triplet.pt"

echo "-> SAGE TRIPLET Semantic Web"
/home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.extract_gcn_embeddings --graphs "$REL_GALLERY" --ckpt "$CKPT_DIR/gcn_encoder_sage_triplet.pth" --out "$OUT_DIR/gcn_cross_test_gallery_sage_triplet.pt"
/home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.extract_gcn_embeddings --graphs "$REL_QUERIES" --ckpt "$CKPT_DIR/gcn_encoder_sage_triplet.pth" --out "$OUT_DIR/gcn_cross_test_queries_sage_triplet.pt"

echo "Estrazione completata in $OUT_DIR"
