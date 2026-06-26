#!/bin/bash
#SBATCH --job-name=pdd_autolabel_pipeline
#SBATCH --output=/homes/kalluri1/Pavement_Distress_Detection/logs/autolabel_pipeline_%j.log
#SBATCH --error=/homes/kalluri1/Pavement_Distress_Detection/logs/autolabel_pipeline_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ksu-gen-gpu.q

# Runs the combined single-command auto_label_pipeline.py on a GPU compute
# node. This MUST go through SLURM -- the head/login node has no GPU, and
# model.track() will fail with "Invalid CUDA device=0 requested" if run
# there directly.
#
# Usage (no file editing needed -- pass the video/folder path as an argument):
#   sbatch scripts/run_autolabel_pipeline.sh /homes/kalluri1/Pavement_Distress_Detection/Dataset/GRMN0015.MP4
#   sbatch scripts/run_autolabel_pipeline.sh /homes/kalluri1/Pavement_Distress_Detection/Dataset/vid2

if [ -z "$1" ]; then
  echo "ERROR: no input path given." >&2
  echo "Usage: sbatch scripts/run_autolabel_pipeline.sh <video_file_or_folder> [--no-gps-ocr]" >&2
  exit 1
fi

INPUT_PATH="$1"
shift  # remaining args ($@), if any (e.g. --no-gps-ocr), are forwarded below

export PATH=~/miniconda3/envs/yolo_env/bin:$PATH

mkdir -p /homes/kalluri1/Pavement_Distress_Detection/logs

echo "Job started on $(hostname) at $(date)"
echo "Input: $INPUT_PATH"
echo "Extra args: $@"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

cd /homes/kalluri1/Pavement_Distress_Detection

~/miniconda3/envs/yolo_env/bin/python -u \
  auto_label_pipeline.py \
  --input "$INPUT_PATH" "$@"

echo "Job finished at $(date)"
