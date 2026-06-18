# Pavement Distress Detection (KDOT-Funded Research)

Automated detection and measurement of pavement distress (cracks, potholes, patches) from dashcam video, using YOLOv11. Developed for Kansas DOT (KDOT) to support maintenance estimation — crack length and pothole quantity — from routine dashcam footage.

- **Researcher:** Deeksha Kalluri, Kansas State University (Project KSU-26-4)
- **Code:** this repository
- **Dataset:** https://huggingface.co/datasets/Deeksha9/pavement-distress-detection
- **Trained models:** https://huggingface.co/Deeksha9/pavement-distress-detection

---

## 1. What this project does

1. Curates and merges pavement distress imagery from 9 public datasets plus original dashcam footage into a unified **6-class** label scheme.
2. Trains YOLOv11 (n/m/x variants) to detect:
   - Longitudinal Crack
   - Transverse Crack
   - Alligator Crack
   - Pothole
   - Patch
   - Block Crack
3. Uses the trained model to **auto-label new, unlabeled dashcam video** — extracting frames, running detection + multi-frame tracking, and keeping only detections that persist across multiple consecutive frames (filtering out one-off false positives like glare or shadows).
4. Periodically retrains on the expanded, auto-labeled dataset ("cycles"), comparing accuracy against the previous cycle.

## 2. Repository structure

```
Pavement_Distress_Detection/
├── scripts/
│   ├── remap_labels.py        # Merge/remap original multi-source labels into 6 final classes
│   ├── split_dataset.py       # 80/10/10 train/val/test split (seed=42)
│   ├── extract_frames.py      # Extract frames from raw dashcam .MP4 at a fixed fps
│   ├── track_and_filter.py    # Auto-label new video: detect + track + filter by persistence
│   ├── train_n.sh / train_m.sh / train_x.sh   # SBATCH training scripts per model size
│   ├── resume_m.sh / resume_x.sh              # Resume training after SLURM time-limit cancellation
│   ├── evaluate.sh            # Run held-out test-set evaluation
│   └── autolabel_vid1.sh      # SBATCH wrapper: extract + track/filter for one video batch
├── Dataset/                   # NOT included in this repo (see Data section below)
├── runs/                      # Training/evaluation outputs — NOT included in this repo
└── auto_labels/                # Auto-labeling cycle outputs — NOT included in this repo
```

**Large files (datasets, trained model weights) are hosted on Hugging Face, not in this Git repository** — GitHub is not suited to hosting tens of thousands of images or multi-hundred-MB model weights. See Section 5.

## 3. Environment setup

This project runs on the Beocat HPC cluster (Kansas State University) using SLURM for job scheduling, but the steps below apply to any Linux machine with SLURM/Conda, with minor path adjustments.

### 3.1 Conda environment

```bash
conda create -n yolo_env python=3.10 -y
conda activate yolo_env
pip install ultralytics opencv-python-headless ffmpeg-python huggingface_hub
```

> **Note:** if your login node's base Python conflicts with the environment (e.g. a `ModuleNotFoundError: No module named 'encodings'` error), always explicitly activate `yolo_env` or call its binaries with full paths (`~/miniconda3/envs/yolo_env/bin/python`, `~/miniconda3/envs/yolo_env/bin/yolo`) rather than relying on a bare `python`/`yolo` on PATH.

### 3.2 Hardware

- Training was run on an NVIDIA L40S GPU (46GB) via SLURM partition `ksu-gen-gpu.q`.
- YOLOv11x training is the most memory-intensive; reduce `batch` size in the training scripts if using a smaller GPU.

## 4. Reproducing the pipeline

### 4.1 Get the data

Download the dataset from Hugging Face (see Section 5 for the link) and place it under `Dataset/` matching the structure referenced in the scripts (`Dataset/full_remapped/`, `Dataset/split/`).

### 4.2 Train a model

```bash
sbatch scripts/train_n.sh   # YOLOv11n (fastest, lowest accuracy)
sbatch scripts/train_m.sh   # YOLOv11m (balanced)
sbatch scripts/train_x.sh   # YOLOv11x (slowest, highest accuracy)
```

Each uses: `imgsz=896`, `cos_lr=True`, full augmentation (mixup, mosaic, HSV jitter, perspective, scale), `patience=30`, `seed=42`. See the scripts for exact hyperparameters.

If a job hits the SLURM time limit before finishing, resume with:

```bash
sbatch scripts/resume_m.sh
sbatch scripts/resume_x.sh
```

### 4.3 Evaluate

```bash
sbatch scripts/evaluate.sh
```

Runs the trained models against the held-out test split and reports mAP50 / mAP50-95 per class.

### 4.4 Auto-label new video

To process a new batch of dashcam video and generate candidate labels for the next training cycle:

1. Place raw `.MP4` files in `Dataset/<batch_name>/` (e.g. `Dataset/vid2/`).
2. In `extract_frames.py` and `track_and_filter.py`, update `VIDEO_DIR`/`FRAMES_DIR` to point at the new batch name. The `BATCH_TAG` mechanism in `extract_frames.py` automatically prefixes output filenames with the batch folder name, so Garmin's recycled video filenames (e.g. `GRMN0015.MP4` reappearing in a later batch) never collide with frames from a previous batch.
3. Submit:
   ```bash
   sbatch scripts/autolabel_vid1.sh
   ```
4. Output lands in `auto_labels/<batch>_cycle<N>/images/` and `.../labels/` — only detections that:
   - Cleared a confidence floor, **and**
   - Persisted across a contiguous run of at least 3 frames (tolerating 1 missed frame), **and**
   - Were sampled down to at most 3 representative frames per persistent detection (to avoid near-duplicate frames of the same physical crack)

   are kept. Everything else (single-frame flickers, low-confidence noise) is discarded automatically.

### 4.5 Retrain on the expanded dataset

Merge the new `auto_labels/<batch>_cycleN/` output into the main dataset, re-run `split_dataset.py`, and repeat Section 4.2-4.3 to measure whether the additional data improved accuracy.

## 5. Data & model hosting (Hugging Face)

Datasets and trained weights are hosted on Hugging Face Hub rather than this Git repository, due to GitHub's 100MB-per-file limit and poor fit for large binary datasets.

| Artifact | Location |
|---|---|
| Full curated dataset (98,662 annotations, pre-split) | `https://huggingface.co/datasets/Deeksha9/pavement-distress-detection` |
| Train/val/test split used for training | same repo, `split/` subfolder |
| Trained weights (YOLOv11 n/m/x) | `https://huggingface.co/Deeksha9/pavement-distress-detection` |

### Downloading the dataset

```bash
pip install huggingface_hub
hf download Deeksha9/pavement-distress-detection --repo-type dataset --local-dir ./Dataset
```

### Downloading trained weights

```bash
hf download Deeksha9/pavement-distress-detection yolov11x_best.pt --local-dir ./weights
hf download Deeksha9/pavement-distress-detection yolov11m_best.pt --local-dir ./weights
hf download Deeksha9/pavement-distress-detection yolov11n_best.pt --local-dir ./weights
```

## 6. Class definitions

| ID | Class | Notes |
|---|---|---|
| 0 | Longitudinal Crack | Runs parallel to direction of travel |
| 1 | Transverse Crack | Runs perpendicular to direction of travel |
| 2 | Alligator Crack | Interconnected crack network |
| 3 | Pothole | |
| 4 | Patch | Prior repair area |
| 5 | Block Crack | Large interconnected rectangular cracking |

This 6-class scheme was simplified from an original 15-class scheme (merged from 9 public datasets) to align with KDOT's maintenance scope (crack sealing, pothole patching, quantity estimation). Classes irrelevant to that scope (e.g. manhole covers) were dropped during remapping.

## 7. Baseline results (test set)

| Class | YOLOv11n mAP50 | YOLOv11m mAP50 | YOLOv11x mAP50 |
|---|---|---|---|
| All | 0.575 | 0.655 | 0.667 |
| Longitudinal Crack | 0.550 | 0.610 | 0.618 |
| Transverse Crack | 0.653 | 0.713 | 0.724 |
| Alligator Crack | 0.613 | 0.676 | 0.682 |
| Pothole | 0.495 | 0.619 | 0.632 |
| Patch | 0.729 | 0.851 | 0.881 |
| Block Crack | 0.407 | 0.460 | 0.465 |

YOLOv11x is used as the production model for auto-labeling new data, given its accuracy advantage outweighs its slower inference (~17ms vs. ~4ms for YOLOv11n) for offline batch processing.

## 8. Contact / attribution

KDOT Project KSU-26-4, Kansas State University. Built on [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) and [ByteTrack](https://github.com/ifzhang/ByteTrack) (via Ultralytics' built-in tracking).
