#!/bin/bash
#SBATCH --job-name=gcn_train
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1 --gres=shard:10000
#SBATCH --time=12:00:00
#SBATCH --output=experiments/logs/gcn-%j.log
#SBATCH --mail-type=END,FAIL
mkdir -p experiments/logs checkpoints

# Ablation Rel vs NoRel: conv default sage.
#   sbatch experiments/jobs/train_cluster.sh sage   # NoRel: solo topologia
#   sbatch experiments/jobs/train_cluster.sh gine   # Rel: usa gli embedding relazioni
# Terzo argomento = dataset (baseline o full)
#   sbatch experiments/jobs/train_cluster.sh gine ntxent baseline
CONV=${1:-sage}
LOSS=${2:-ntxent}
DATA_MODE=${3:-full}

# checkpoint distinto, per non sovrascrivere
SUFFIX=""; [ "$LOSS" != "ntxent" ] && SUFFIX="_${LOSS}"
DATA_PREFIX=""; [ "$DATA_MODE" == "baseline" ] && DATA_PREFIX="baseline_"

if [ "$DATA_MODE" == "baseline" ]; then
    TRAIN_DATA="data/sceneGraph/embedded/full_set_baseline/train_scene_graphs.pt"
else
    TRAIN_DATA="data/sceneGraph/embedded/full_set/train_scene_graphs.pt"
fi

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
        --out "checkpoints/${DATA_PREFIX}gcn_encoder_${CONV}${SUFFIX}.pth"
