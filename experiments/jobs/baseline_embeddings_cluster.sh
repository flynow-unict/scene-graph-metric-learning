#!/bin/bash
#SBATCH --job-name=embed_baseline
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:20000
#SBATCH --time=02:00:00
#SBATCH --output=experiments/logs/nomic-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=TUO_EMAIL@studium.unict.it

echo "========================================="
echo "Avvio calcolo embeddings per il Baseline Set su nodo SLURM"
echo "========================================="

# Creiamo la cartella dei log se non esiste
mkdir -p experiments/logs

# Eseguiamo lo script Python all'interno del container Apptainer fornito dal cluster
# Il parametro --nv serve per abilitare la GPU Nvidia dentro il container
apptainer run --nv /shared/sifs/latest.sif python -u -m src.datasets.add_baseline_nomic_embeddings

echo "========================================="
echo "Elaborazione completata!"
echo "========================================="
