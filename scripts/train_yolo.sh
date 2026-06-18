#!/bin/bash
#SBATCH --job-name=pdd_train
#SBATCH --output=/homes/kalluri1/Pavement_Distress_Detection/logs/train_%j.log
#SBATCH --error=/homes/kalluri1/Pavement_Distress_Detection/logs/train_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ksu-gen-gpu.q

source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolo_env
export PATH=~/miniconda3/envs/yolo_env/bin:$PATH

DATA=/homes/kalluri1/Pavement_Distress_Detection/Dataset/split/data.yaml
OUT=/homes/kalluri1/Pavement_Distress_Detection/runs

mkdir -p /homes/kalluri1/Pavement_Distress_Detection/logs

echo "========================================="
echo "Starting PDD Training - $(date)"
echo "========================================="

echo "Training YOLOv11n..."
~/miniconda3/envs/yolo_env/bin/yolo detect train model=yolo11n.pt data=$DATA epochs=100 imgsz=640 batch=32 project=$OUT name=yolo11n patience=20 workers=8 device=0 exist_ok=True

echo "Training YOLOv11m..."
~/miniconda3/envs/yolo_env/bin/yolo detect train model=yolo11m.pt data=$DATA epochs=100 imgsz=640 batch=16 project=$OUT name=yolo11m patience=20 workers=8 device=0 exist_ok=True

echo "Training YOLOv11x..."
~/miniconda3/envs/yolo_env/bin/yolo detect train model=yolo11x.pt data=$DATA epochs=100 imgsz=640 batch=8 project=$OUT name=yolo11x patience=20 workers=8 device=0 exist_ok=True

echo "========================================="
echo "All training complete - $(date)"
echo "========================================="
