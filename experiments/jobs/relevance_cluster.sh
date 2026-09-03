#!/bin/bash
#SBATCH --job-name=relevance_build
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:20000
#SBATCH --time=02:00:00
#SBATCH --output=experiments/logs/relevance-%j.log
#SBATCH --mail-type=END,FAIL

# Utilizzo:
# sbatch experiments/jobs/relevance_cluster.sh baseline
# sbatch experiments/jobs/relevance_cluster.sh full

DATA_MODE=${1:-full}

if [ "$DATA_MODE" == "baseline" ]; then
    QUERIES="data/sceneGraph/embedded/full_set_baseline/test_queries_scene_graphs.pt"
    GALLERY="data/sceneGraph/embedded/full_set_baseline/test_gallery_scene_graphs.pt"
    OUT="data/gcn/relevance_test_baseline.pt"
else
    QUERIES="data/sceneGraph/embedded/full_set/test_queries_scene_graphs.pt"
    GALLERY="data/sceneGraph/embedded/full_set/test_gallery_scene_graphs.pt"
    OUT="data/gcn/relevance_test_full.pt"
fi

echo "========================================="
echo "Avvio calcolo Ground Truth (Relevance) su nodo SLURM"
echo "Mode: $DATA_MODE"
echo "Output: $OUT"
echo "========================================="

mkdir -p experiments/logs data/gcn

SIF=/shared/sifs/latest.sif

export PYTHONUSERBASE=$HOME/.pyg-user
apptainer run $SIF python -c "import torch_geometric" 2>/dev/null || \
    apptainer run $SIF python -m pip install --user torch_geometric

apptainer run --nv $SIF \
    python -u -m src.evaluation.build_relevance \
        --queries "$QUERIES" \
        --gallery "$GALLERY" \
        --out "$OUT"

echo "========================================="
echo "Calcolo completato!"
echo "========================================="
