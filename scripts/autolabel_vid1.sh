#!/bin/bash
#SBATCH --job-name=pdd_autolabel
#SBATCH --output=/homes/kalluri1/Pavement_Distress_Detection/logs/autolabel_%j.log
#SBATCH --error=/homes/kalluri1/Pavement_Distress_Detection/logs/autolabel_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ksu-gen-gpu.q

export PATH=~/miniconda3/envs/yolo_env/bin:$PATH

echo "============================================"
echo "Step 1: Extracting frames - $(date)"
echo "============================================"

~/miniconda3/envs/yolo_env/bin/python -u \
    ~/Pavement_Distress_Detection/scripts/extract_frames.py

echo "============================================"
echo "Step 2: Tracking and filtering frames - $(date)"
echo "============================================"

~/miniconda3/envs/yolo_env/bin/python -u \
    ~/Pavement_Distress_Detection/scripts/track_and_filter.py

echo "============================================"
echo "Done - $(date)"
echo "============================================"
