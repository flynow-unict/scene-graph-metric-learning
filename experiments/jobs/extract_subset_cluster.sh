#!/bin/bash
#SBATCH --job-name=gcn_extract_subset
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1 --gres=shard:10000
#SBATCH --time=02:00:00
#SBATCH --output=experiments/logs/extract-subset-%j.log
#SBATCH --mail-type=END,FAIL

# Utilizzo:
# sbatch experiments/jobs/extract_subset_cluster.sh gine ntxent baseline
# sbatch experiments/jobs/extract_subset_cluster.sh sage triplet semantic
CONV=${1:-sage}
LOSS=${2:-ntxent}
DATA_MODE=${3:-baseline}  # 'baseline' o 'semantic'

SUFFIX=""; [ "$LOSS" != "ntxent" ] && SUFFIX="_${LOSS}"

if [ "$DATA_MODE" == "baseline" ]; then
    QUERIES="data/sceneGraph/subset/baseline/embedded/test_queries_scene_graphs.pt"
    GALLERY="data/sceneGraph/subset/baseline/embedded/test_gallery_scene_graphs.pt"
    CKPT="checkpoints/subset/checkpoints/baseline_gcn_encoder_${CONV}${SUFFIX}.pth"
    OUT_Q="data/models/subset/baseline/gcn/baseline_gcn_test_queries_${CONV}${SUFFIX}.pt"
    OUT_G="data/models/subset/baseline/gcn/baseline_gcn_test_gallery_${CONV}${SUFFIX}.pt"
else
    QUERIES="data/sceneGraph/subset/semantic/embedded/test_queries_scene_graphs.pt"
    GALLERY="data/sceneGraph/subset/semantic/embedded/test_gallery_scene_graphs.pt"
    CKPT="checkpoints/subset/checkpoints/gcn_encoder_${CONV}${SUFFIX}.pth"
    # Per ricalcare la struttura esistente sul subset semantic_web:
    # es: gcn_gine_test_gallery.pt / gcn_gine_test_queries.pt
    M_NAME="${CONV}${SUFFIX}"
    OUT_Q="data/models/subset/semantic_web/gcn/gcn_${M_NAME}_test_queries.pt"
    OUT_G="data/models/subset/semantic_web/gcn/gcn_${M_NAME}_test_gallery.pt"
fi

echo "========================================="
echo "Avvio estrazione embedding GCN per Sotto-Dataset"
echo "Checkpoint: $CKPT"
echo "Dataset Mode: $DATA_MODE"
echo "Query Output: $OUT_Q"
echo "Gallery Output: $OUT_G"
echo "========================================="

mkdir -p experiments/logs
mkdir -p "$(dirname "$OUT_Q")"
mkdir -p "$(dirname "$OUT_G")"

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
