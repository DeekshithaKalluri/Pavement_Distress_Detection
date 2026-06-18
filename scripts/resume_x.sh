#!/bin/bash
#SBATCH --job-name=pdd_x_resume
#SBATCH --output=/homes/kalluri1/Pavement_Distress_Detection/logs/resume_x_%j.log
#SBATCH --error=/homes/kalluri1/Pavement_Distress_Detection/logs/resume_x_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ksu-gen-gpu.q

source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolo_env
export PATH=~/miniconda3/envs/yolo_env/bin:$PATH

echo "Resuming YOLOv11x from epoch 49 - $(date)"

~/miniconda3/envs/yolo_env/bin/yolo detect train \
  resume=True \
  model=/homes/kalluri1/Pavement_Distress_Detection/runs/train/yolov11x/x.baseline/weights/last.pt \
  workers=4 \
  device=0

echo "YOLOv11x resume complete - $(date)"
