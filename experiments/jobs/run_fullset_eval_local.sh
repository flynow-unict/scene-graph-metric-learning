#!/bin/bash
set -e

echo "========================================="
echo " FASE 1: Costruzione Indici FAISS "
echo "========================================="
/home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.build_fullset_faiss_index

echo ""
echo "========================================="
echo " FASE 2: Calcolo Metriche (MRR, R@K) "
echo "========================================="
/home/santi/anaconda3/envs/machineLearning/bin/python -m src.evaluation.evaluate_fullset_models

echo ""
echo "Il benchmark è stato calcolato e salvato nella cartella 'results'!"
