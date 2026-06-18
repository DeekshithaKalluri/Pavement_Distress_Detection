#!/bin/bash
#SBATCH --job-name=pdd_eval
#SBATCH --output=/homes/kalluri1/Pavement_Distress_Detection/logs/eval_%j.log
#SBATCH --error=/homes/kalluri1/Pavement_Distress_Detection/logs/eval_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=ksu-gen-gpu.q

source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolo_env
export PATH=~/miniconda3/envs/yolo_env/bin:$PATH

DATA=/homes/kalluri1/Pavement_Distress_Detection/Dataset/split/data.yaml
RUNS=/homes/kalluri1/Pavement_Distress_Detection/runs

echo "========================================="
echo "Evaluating all models on test set - $(date)"
echo "========================================="

echo "--- YOLOv11n ---"
~/miniconda3/envs/yolo_env/bin/yolo detect val \
  model=$RUNS/train/yolov11n/n.baseline/weights/best.pt \
  data=$DATA \
  split=test \
  workers=4 \
  device=0 \
  name=eval_n \
  project=$RUNS/eval \
  exist_ok=True

echo "--- YOLOv11m ---"
~/miniconda3/envs/yolo_env/bin/yolo detect val \
  model=$RUNS/train/yolov11m/m.baseline/weights/best.pt \
  data=$DATA \
  split=test \
  workers=4 \
  device=0 \
  name=eval_m \
  project=$RUNS/eval \
  exist_ok=True

echo "--- YOLOv11x ---"
~/miniconda3/envs/yolo_env/bin/yolo detect val \
  model=$RUNS/train/yolov11x/x.baseline/weights/best.pt \
  data=$DATA \
  split=test \
  workers=4 \
  device=0 \
  name=eval_x \
  project=$RUNS/eval \
  exist_ok=True

echo "========================================="
echo "Evaluation complete - $(date)"
echo "========================================="
