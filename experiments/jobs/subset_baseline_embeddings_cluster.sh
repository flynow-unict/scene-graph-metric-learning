#!/bin/bash
#SBATCH --job-name=embed_subset_baseline
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:20000
#SBATCH --time=01:00:00
#SBATCH --output=experiments/logs/nomic-subset-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=TUO_EMAIL@studium.unict.it

echo "========================================="
echo "Avvio calcolo embeddings per il Subset Baseline su nodo SLURM"
echo "========================================="

mkdir -p experiments/logs

SPLITS=("train_scene_graphs.pt" "val_scene_graphs.pt" "test_gallery_scene_graphs.pt" "test_queries_scene_graphs.pt")

for SPLIT in "${SPLITS[@]}"; do
    echo "Elaborazione $SPLIT..."
    INPUT_FILE="data/sceneGraph/subset/raw/subset_baseline/$SPLIT"
    OUTPUT_FILE="data/sceneGraph/subset/embedded/subset_baseline/$SPLIT"
    
    apptainer run --nv /shared/sifs/latest.sif python -u -m src.datasets.add_nomic_embeddings --input "$INPUT_FILE" --output "$OUTPUT_FILE"
done

echo "========================================="
echo "Elaborazione completata!"
echo "========================================="
