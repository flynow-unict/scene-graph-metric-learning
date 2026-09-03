#!/bin/bash
#SBATCH --job-name=gcn_train_subset
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1 --gres=shard:10000
#SBATCH --time=12:00:00
#SBATCH --output=experiments/logs/gcn-subset-%j.log
#SBATCH --mail-type=END,FAIL

# Utilizzo:
# sbatch experiments/jobs/train_subset_cluster.sh sage ntxent baseline
# sbatch experiments/jobs/train_subset_cluster.sh gine triplet baseline
CONV=${1:-sage}
LOSS=${2:-ntxent}
DATA_MODE=${3:-baseline}  # 'baseline' o 'semantic'

# checkpoint distinto, per non sovrascrivere
SUFFIX=""; [ "$LOSS" != "ntxent" ] && SUFFIX="_${LOSS}"
DATA_PREFIX=""; [ "$DATA_MODE" == "baseline" ] && DATA_PREFIX="baseline_"

if [ "$DATA_MODE" == "baseline" ]; then
    TRAIN_DATA="data/sceneGraph/subset/baseline/embedded/train_scene_graphs.pt"
else
    TRAIN_DATA="data/sceneGraph/subset/semantic/embedded/train_scene_graphs.pt"
fi

OUT_CKPT="checkpoints/subset/checkpoints/${DATA_PREFIX}gcn_encoder_${CONV}${SUFFIX}.pth"

echo "========================================="
echo "Avvio addestramento GCN per Sotto-Dataset"
echo "Architettura: $CONV"
echo "Loss: $LOSS"
echo "Dati: $TRAIN_DATA"
echo "Checkpoint Output: $OUT_CKPT"
echo "========================================="

mkdir -p experiments/logs checkpoints/subset/checkpoints

SIF=/shared/sifs/latest.sif

export PYTHONUSERBASE=$HOME/.pyg-user
apptainer run $SIF python -c "import torch_geometric" 2>/dev/null || \
    apptainer run $SIF python -m pip install --user torch_geometric

apptainer run --nv $SIF \
    python -u -m src.training.train \
        --data "$TRAIN_DATA" \
        --epochs 300 \
        --batch-size 128 \
        --conv "$CONV" \
        --loss "$LOSS" \
        --positives mixed \
        --temperature 0.2 \
        --out "$OUT_CKPT"
