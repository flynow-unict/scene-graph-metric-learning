#!/bin/bash
#SBATCH --job-name=gcn_extract
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1 --gres=shard:10000
#SBATCH --time=02:00:00
#SBATCH --output=experiments/logs/extract-%j.log
#SBATCH --mail-type=END,FAIL

# Utilizzo:
# sbatch experiments/jobs/extract_cluster.sh gine ntxent baseline
CONV=${1:-sage}
LOSS=${2:-ntxent}
DATA_MODE=${3:-full}

SUFFIX=""; [ "$LOSS" != "ntxent" ] && SUFFIX="_${LOSS}"
DATA_PREFIX=""; [ "$DATA_MODE" == "baseline" ] && DATA_PREFIX="baseline_"

if [ "$DATA_MODE" == "baseline" ]; then
    QUERIES="data/sceneGraph/embedded/full_set_baseline/test_queries_scene_graphs.pt"
    GALLERY="data/sceneGraph/embedded/full_set_baseline/test_gallery_scene_graphs.pt"
else
    QUERIES="data/sceneGraph/embedded/full_set/test_queries_scene_graphs.pt"
    GALLERY="data/sceneGraph/embedded/full_set/test_gallery_scene_graphs.pt"
fi

CKPT="checkpoints/${DATA_PREFIX}gcn_encoder_${CONV}${SUFFIX}.pth"
OUT_Q="data/gcn/${DATA_PREFIX}gcn_test_queries_${CONV}${SUFFIX}.pt"
OUT_G="data/gcn/${DATA_PREFIX}gcn_test_gallery_${CONV}${SUFFIX}.pt"

echo "========================================="
echo "Avvio estrazione embedding GCN"
echo "Checkpoint: $CKPT"
echo "Dataset Mode: $DATA_MODE"
echo "========================================="

SIF=/shared/sifs/latest.sif
export PYTHONUSERBASE=$HOME/.pyg-user

apptainer run --nv $SIF \
    python -u -m src.evaluation.extract_gcn_embeddings \
        --split test_queries \
        --graphs "$QUERIES" \
        --ckpt "$CKPT" \
        --out "$OUT_Q"

apptainer run --nv $SIF \
    python -u -m src.evaluation.extract_gcn_embeddings \
        --split test_gallery \
        --graphs "$GALLERY" \
        --ckpt "$CKPT" \
        --out "$OUT_G"

echo "========================================="
echo "Estrazione completata!"
echo "========================================="
