#!/bin/bash
#SBATCH --job-name=pdd_x_cycle1_resume
#SBATCH --output=/homes/kalluri1/Pavement_Distress_Detection/logs/train_x_cycle1_resume_%j.log
#SBATCH --error=/homes/kalluri1/Pavement_Distress_Detection/logs/train_x_cycle1_resume_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ksu-gen-gpu.q

# Resumes the cycle-1 YOLOv11x training run from its last saved checkpoint,
# if the original job (train_x_cycle1.sh) was cut off by the 48-hour SLURM
# time limit before completing all 100 epochs.
#
# Ultralytics saves last.pt after every completed epoch into the run folder,
# specifically to support exactly this kind of resume. Passing resume=True
# together with the SAME run folder (model=.../last.pt) picks training back
# up from the next epoch, preserving optimizer state, learning-rate schedule
# position, and training history -- this is not the same as starting a new
# 100-epoch run from scratch.
#
# Only run this AFTER confirming the original job has actually stopped:
#   squeue -u kalluri1
# If it's still listed as running, there is nothing to resume yet.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolo_env
export PATH=~/miniconda3/envs/yolo_env/bin:$PATH

LAST_CHECKPOINT=/homes/kalluri1/Pavement_Distress_Detection/runs/train/yolov11x/x.cycle1/weights/last.pt

if [ ! -f "$LAST_CHECKPOINT" ]; then
  echo "ERROR: no checkpoint found at $LAST_CHECKPOINT"
  echo "Nothing to resume -- check the original run folder before retrying."
  exit 1
fi

echo "Resuming YOLOv11x cycle1 from $LAST_CHECKPOINT - $(date)"

~/miniconda3/envs/yolo_env/bin/yolo detect train \
  resume=True \
  model="$LAST_CHECKPOINT" \
  device=0

echo "YOLOv11x cycle1 resume run finished - $(date)"
