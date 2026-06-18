#!/bin/bash
#SBATCH --job-name=create_env
#SBATCH --output=/homes/kalluri1/Pavement_Distress_Detection/logs/create_env_%j.log
#SBATCH --error=/homes/kalluri1/Pavement_Distress_Detection/logs/create_env_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=ksu-gen-gpu.q
#SBATCH --gres=gpu:1

source ~/miniconda3/etc/profile.d/conda.sh

echo "Creating video_env..."
conda create -n video_env python=3.10 -y
echo "Create done: $?"

echo "Installing ffmpeg..."
conda install -n video_env -c conda-forge ffmpeg -y
echo "ffmpeg install done: $?"

echo "Installing opencv..."
conda install -n video_env -c conda-forge opencv -y
echo "opencv install done: $?"

echo "Verifying..."
ls ~/miniconda3/envs/video_env/bin/ffmpeg
~/miniconda3/envs/video_env/bin/ffmpeg -version | head -1
~/miniconda3/envs/video_env/bin/python -c "import cv2; print('OpenCV:', cv2.__version__)"
echo "All done!"
