#!/bin/bash
#SBATCH --job-name=raw_track_sample
#SBATCH --output=/homes/kalluri1/Pavement_Distress_Detection/logs/raw_track_sample_%j.log
#SBATCH --error=/homes/kalluri1/Pavement_Distress_Detection/logs/raw_track_sample_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ksu-gen-gpu.q

# Diagnostic-only job: runs raw (unfiltered) tracking on a handful of
# sampled videos to check whether rare classes (Block Crack, Patch,
# Alligator Crack) are ever proposed by the model at all in this footage,
# before any persistence filtering is applied.
#
# Mirrors the GPU/partition settings used by autolabel_vid1.sh -- this
# MUST run on a GPU compute node via SLURM, not interactively on the
# login/head node (the head node has no GPU and model.track() will fail
# silently/early if attempted there).

set -e

export PATH=~/miniconda3/envs/yolo_env/bin:$PATH

mkdir -p /homes/kalluri1/Pavement_Distress_Detection/logs

echo "Job started on $(hostname) at $(date)"
echo "GPU visible to this job:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

cd /homes/kalluri1/Pavement_Distress_Detection

~/miniconda3/envs/yolo_env/bin/python -u \
  scripts/raw_tracking_sample.py \
  --weights /homes/kalluri1/Pavement_Distress_Detection/runs/train/yolov11x/x.baseline/weights/best.pt \
  --frames-root /homes/kalluri1/Pavement_Distress_Detection/Dataset/frames/vid1 \
  --out /homes/kalluri1/Pavement_Distress_Detection/auto_labels/raw_sample_check \
  --num-videos 5

echo "Job finished at $(date)"
