#!/bin/bash
#SBATCH --job-name=pdd_x_cycle1
#SBATCH --output=/homes/kalluri1/Pavement_Distress_Detection/logs/train_x_cycle1_%j.log
#SBATCH --error=/homes/kalluri1/Pavement_Distress_Detection/logs/train_x_cycle1_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ksu-gen-gpu.q

# Cycle 1 retrain: identical settings to train_x.sh (the original baseline
# run), except pointing at Dataset/split_cycle1/data.yaml (baseline +
# vid1_cycle1 auto-labels merged and re-split) instead of the original
# Dataset/split/data.yaml, and writing to a separate run name/folder so the
# original x.baseline run is preserved untouched for comparison.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolo_env
export PATH=~/miniconda3/envs/yolo_env/bin:$PATH

mkdir -p /homes/kalluri1/Pavement_Distress_Detection/runs/train/yolov11x/x.cycle1

echo "Starting YOLOv11x cycle1 - $(date)"

~/miniconda3/envs/yolo_env/bin/yolo detect train \
  data=/homes/kalluri1/Pavement_Distress_Detection/Dataset/split_cycle1/data.yaml \
  model=yolo11x.pt \
  epochs=100 \
  imgsz=896 \
  batch=8 \
  patience=30 \
  cos_lr=True \
  augment=True \
  mixup=0.1 \
  mosaic=1.0 \
  hsv_h=0.015 \
  hsv_s=0.7 \
  hsv_v=0.4 \
  perspective=0.001 \
  scale=0.8 \
  seed=42 \
  device=0 \
  name=x.cycle1 \
  project=/homes/kalluri1/Pavement_Distress_Detection/runs/train/yolov11x \
  exist_ok=True

echo "YOLOv11x cycle1 complete - $(date)"
