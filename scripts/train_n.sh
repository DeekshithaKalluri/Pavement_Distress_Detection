#!/bin/bash
#SBATCH --job-name=pdd_n
#SBATCH --output=/homes/kalluri1/Pavement_Distress_Detection/logs/train_n_%j.log
#SBATCH --error=/homes/kalluri1/Pavement_Distress_Detection/logs/train_n_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ksu-gen-gpu.q

source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolo_env
export PATH=~/miniconda3/envs/yolo_env/bin:$PATH

mkdir -p /homes/kalluri1/Pavement_Distress_Detection/runs/train/yolov11n/n.baseline

echo "Starting YOLOv11n - $(date)"

~/miniconda3/envs/yolo_env/bin/yolo detect train \
  data=/homes/kalluri1/Pavement_Distress_Detection/Dataset/split/data.yaml \
  model=yolo11n.pt \
  epochs=100 \
  imgsz=896 \
  batch=16 \
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
  name=n.baseline \
  project=/homes/kalluri1/Pavement_Distress_Detection/runs/train/yolov11n \
  exist_ok=True

echo "YOLOv11n complete - $(date)"
