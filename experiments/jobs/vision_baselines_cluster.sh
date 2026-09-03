#!/bin/bash
#SBATCH --job-name=extract_vision
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1 --gres=shard:10000
#SBATCH --time=04:00:00
#SBATCH --output=experiments/logs/vision-%j.log
#SBATCH --mail-type=END,FAIL

SIF=/shared/sifs/latest.sif
export PYTHONUSERBASE=$HOME/.pyg-user

# Uso come riferimento i grafi della baseline (le immagini sono le stesse)
QUERIES="data/sceneGraph/embedded/full_set_baseline/test_queries_scene_graphs.pt"
GALLERY="data/sceneGraph/embedded/full_set_baseline/test_gallery_scene_graphs.pt"
IMG_DIR_Q="data/images/fullset/Test"
IMG_DIR_G="data/images/fullset/VectorDB"
OUT_DIR="data/models/fullset/vision_baselines"

mkdir -p $OUT_DIR

echo "=== Estrazione CLIP ==="
apptainer run --nv $SIF python -u -m src.evaluation.extract_baseline_embeddings \
    --encoder clip --graphs "$QUERIES" --images-dir "$IMG_DIR_Q" --out "$OUT_DIR/clip_test_queries.pt"
apptainer run --nv $SIF python -u -m src.evaluation.extract_baseline_embeddings \
    --encoder clip --graphs "$GALLERY" --images-dir "$IMG_DIR_G" --out "$OUT_DIR/clip_test_gallery.pt"

echo "=== Estrazione ResNet ==="
apptainer run --nv $SIF python -u -m src.evaluation.extract_baseline_embeddings \
    --encoder resnet --graphs "$QUERIES" --images-dir "$IMG_DIR_Q" --out "$OUT_DIR/resnet_test_queries.pt"
apptainer run --nv $SIF python -u -m src.evaluation.extract_baseline_embeddings \
    --encoder resnet --graphs "$GALLERY" --images-dir "$IMG_DIR_G" --out "$OUT_DIR/resnet_test_gallery.pt"

echo "Fatto!"
