#!/bin/bash
#SBATCH --job-name=embeddings_nomic
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-medium
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1 --gres=shard:5000
#SBATCH --time=02:00:00
#SBATCH --output=logs/job-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=TUO_EMAIL@studium.unict.it

echo "========================================="
echo "Avvio calcolo embeddings su nodo SLURM"
echo "========================================="

# Adattato per supportare inferenza online o batch
JOB_ID=${1:-"batch"}

if [ "$JOB_ID" != "batch" ]; then
    echo "Modalità Inferenza Singola - Job ID: $JOB_ID"
    INPUT_PT="data/sceneGraph/Raw/inference/${JOB_ID}.pt"
    OUTPUT_DIR="data/sceneGraph/embedded/inference"
    mkdir -p "$OUTPUT_DIR"
    OUTPUT_PT="${OUTPUT_DIR}/${JOB_ID}.pt"
    
    export HF_HUB_OFFLINE=1
    apptainer run --nv /shared/sifs/latest.sif python script/dataLoader/add_nomic_embeddings.py --input_pt "$INPUT_PT" --output_pt "$OUTPUT_PT"
else
    echo "Modalità Batch Offline"
    export HF_HUB_OFFLINE=1
    apptainer run --nv /shared/sifs/latest.sif python script/dataLoader/add_nomic_embeddings.py
fi

echo "========================================="
echo "Elaborazione completata!"
echo "========================================="
