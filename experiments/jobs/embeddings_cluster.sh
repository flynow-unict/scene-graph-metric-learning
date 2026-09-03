#!/bin/bash
#SBATCH --job-name=embeddings_nomic
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-medium
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1 --gres=shard:5000
#SBATCH --time=02:00:00
#SBATCH --output=experiments/logs/job-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=TUO_EMAIL@studium.unict.it

echo "========================================="
echo "Avvio calcolo embeddings su nodo SLURM"
echo "========================================="

# Export del proxy di rete del cluster UNICT
export http_proxy="http://${USER}:x@proxy:3128"
export https_proxy="http://${USER}:x@proxy:3128"
export HTTP_PROXY="http://${USER}:x@proxy:3128"
export HTTPS_PROXY="http://${USER}:x@proxy:3128"

export APPTAINERENV_http_proxy="$http_proxy"
export APPTAINERENV_https_proxy="$https_proxy"

JOB_ID=${1}

if [ -n "$JOB_ID" ]; then
    INPUT_FILE="data/sceneGraph/Raw/inference/${JOB_ID}.pt"
    OUTPUT_FILE="data/sceneGraph/embedded/inference/${JOB_ID}.pt"
    echo "Processing single inference job: $JOB_ID"
    echo "Input: $INPUT_FILE"
    echo "Output: $OUTPUT_FILE"
    apptainer run --nv /shared/sifs/latest.sif python -m src.datasets.add_nomic_embeddings --input "$INPUT_FILE" --output "$OUTPUT_FILE"
else
    echo "Processing default batch dataset..."
    apptainer run --nv /shared/sifs/latest.sif python -m src.datasets.add_nomic_embeddings
fi

echo "========================================="
echo "Elaborazione completata!"
echo "========================================="
