#!/bin/bash
#SBATCH --job-name=vlm_extractor
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:20000
#SBATCH --time=01:00:00
#SBATCH --output=experiments/logs/vlm-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=TUO_EMAIL@studium.unict.it

echo "========================================="
echo "Avvio estrazione VLM (LLaVA) su nodo GPU"
echo "========================================="

# Export del proxy di rete del cluster UNICT per consentire ai nodi GPU di scaricare da HuggingFace/Ollama
export http_proxy="http://${USER}:x@proxy:3128"
export https_proxy="http://${USER}:x@proxy:3128"
export HTTP_PROXY="http://${USER}:x@proxy:3128"
export HTTPS_PROXY="http://${USER}:x@proxy:3128"

export APPTAINERENV_http_proxy="$http_proxy"
export APPTAINERENV_https_proxy="$https_proxy"

# MODEL_ID deve corrispondere esattamente al nome del repository HuggingFace.
MODEL_ID=${1:-"Qwen/Qwen2.5-VL-7B-Instruct"}
IMAGE_PATH=${2:-"data/images/images_all_15k/236.jpg"}
JOB_ID=${3:-"inference_job"}

OUTPUT_PT="data/sceneGraph/Raw/inference/${JOB_ID}.pt"

echo "Modello selezionato: $MODEL_ID"
echo "Immagine target: $IMAGE_PATH"
echo "Job ID: $JOB_ID"
echo "Output PT: $OUTPUT_PT"

echo "Avvio estrazione..."
apptainer run --nv /shared/sifs/latest.sif python -m src.datasets.vlm_extractor --image "$IMAGE_PATH" --mode cluster --model "$MODEL_ID" --output_pt "$OUTPUT_PT" --job_id "$JOB_ID"

echo "========================================="
echo "Estrazione VLM completata!"
echo "========================================="
