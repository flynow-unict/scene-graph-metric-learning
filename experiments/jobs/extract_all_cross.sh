#!/bin/bash
set -e

PYTHON="/home/santi/anaconda3/envs/machineLearning/bin/python"
EXTRACT="-m src.evaluation.extract_gcn_embeddings"

OUT_CROSS="data/models/cross_evaluation"
OUT_REV="data/models/reverse_cross_evaluation"

mkdir -p "$OUT_CROSS" "$OUT_REV"

# ==========================================
# FASE 1: CROSS EVALUATION (Modelli FULLSET -> Dati SUBSET)
# ==========================================
echo "Inizio CROSS EVALUATION (Modelli Fullset su Dati Subset)..."

# Grafi Subset
SUB_SEM_Q="data/sceneGraph/subset/semantic/embedded/test_queries_scene_graphs.pt"
SUB_SEM_G="data/sceneGraph/subset/semantic/embedded/test_gallery_scene_graphs.pt"
SUB_BASE_Q="data/sceneGraph/subset/baseline/embedded/test_queries_scene_graphs.pt"
SUB_BASE_G="data/sceneGraph/subset/baseline/embedded/test_gallery_scene_graphs.pt"

# Checkpoint Fullset
CKPT_FULL="data/models/fullset/checkpoints"

# 1.1 Fullset Semantic Web (usa Subset Semantic Graphs)
for arch in "gine" "gine_triplet" "sage" "sage_triplet"; do
    echo "[Cross] Fullset Semantic -> Subset Graph: $arch"
    $PYTHON $EXTRACT --split test_queries --graphs "$SUB_SEM_Q" --ckpt "$CKPT_FULL/gcn_encoder_${arch}.pth" --out "$OUT_CROSS/semantic_test_queries_${arch}.pt"
    $PYTHON $EXTRACT --split test_gallery --graphs "$SUB_SEM_G" --ckpt "$CKPT_FULL/gcn_encoder_${arch}.pth" --out "$OUT_CROSS/semantic_test_gallery_${arch}.pt"
done

# 1.2 Fullset Baseline (usa Subset Baseline Graphs)
for arch in "gine" "gine_triplet" "sage" "sage_triplet"; do
    echo "[Cross] Fullset Baseline -> Subset Graph: $arch"
    $PYTHON $EXTRACT --split test_queries --graphs "$SUB_BASE_Q" --ckpt "$CKPT_FULL/baseline_gcn_encoder_${arch}.pth" --out "$OUT_CROSS/baseline_test_queries_${arch}.pt"
    $PYTHON $EXTRACT --split test_gallery --graphs "$SUB_BASE_G" --ckpt "$CKPT_FULL/baseline_gcn_encoder_${arch}.pth" --out "$OUT_CROSS/baseline_test_gallery_${arch}.pt"
done


# ==========================================
# FASE 2: REVERSE CROSS EVALUATION (Modelli SUBSET -> Dati FULLSET)
# ==========================================
echo "Inizio REVERSE CROSS EVALUATION (Modelli Subset su Dati Fullset)..."

# Grafi Fullset
FULL_SEM_Q="data/sceneGraph/fullset/semantic/embedded/test_queries_scene_graphs.pt"
FULL_SEM_G="data/sceneGraph/fullset/semantic/embedded/test_gallery_scene_graphs.pt"
FULL_BASE_Q="data/sceneGraph/fullset/baseline/embedded/test_queries_scene_graphs.pt"
FULL_BASE_G="data/sceneGraph/fullset/baseline/embedded/test_gallery_scene_graphs.pt"

# Checkpoint Subset
CKPT_SUB="data/models/subset/checkpoints"

# 2.1 Subset Semantic Web (usa Fullset Semantic Graphs)
for arch in "gine" "gine_triplet" "sage" "sage_triplet"; do
    echo "[RevCross] Subset Semantic -> Fullset Graph: $arch"
    $PYTHON $EXTRACT --split test_queries --graphs "$FULL_SEM_Q" --ckpt "$CKPT_SUB/gcn_encoder_${arch}.pth" --out "$OUT_REV/semantic_test_queries_${arch}.pt"
    $PYTHON $EXTRACT --split test_gallery --graphs "$FULL_SEM_G" --ckpt "$CKPT_SUB/gcn_encoder_${arch}.pth" --out "$OUT_REV/semantic_test_gallery_${arch}.pt"
done

# 2.2 Subset Baseline (usa Fullset Baseline Graphs)
for arch in "gine" "gine_triplet" "sage" "sage_triplet"; do
    echo "[RevCross] Subset Baseline -> Fullset Graph: $arch"
    $PYTHON $EXTRACT --split test_queries --graphs "$FULL_BASE_Q" --ckpt "$CKPT_SUB/baseline_gcn_encoder_${arch}.pth" --out "$OUT_REV/baseline_test_queries_${arch}.pt"
    $PYTHON $EXTRACT --split test_gallery --graphs "$FULL_BASE_G" --ckpt "$CKPT_SUB/baseline_gcn_encoder_${arch}.pth" --out "$OUT_REV/baseline_test_gallery_${arch}.pt"
done

echo "TUTTE LE ESTRAZIONI COMPLETA!"
