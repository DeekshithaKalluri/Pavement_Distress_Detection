"""
sample_for_review.py

Randomly samples a set of kept auto-labeled images for manual visual review,
primarily to check for the known lane-line-mistaken-for-longitudinal-crack
false positive pattern before merging a new auto-label cycle into the
training set.

Specifically oversamples class 0 (Longitudinal Crack) detections, since
that's the class affected by the known lane-line issue, while still
including a baseline random sample across all classes so other failure
modes aren't missed.

Usage:
    python sample_for_review.py \
        --images ~/Pavement_Distress_Detection/auto_labels/vid1_cycle1/images \
        --labels ~/Pavement_Distress_Detection/auto_labels/vid1_cycle1/labels \
        --out ~/Pavement_Distress_Detection/review_sample_cycle1 \
        --total 50 \
        --target-class 0 \
        --target-fraction 0.6

This copies (not symlinks, so the sample is self-contained and portable)
the sampled images plus a single review_checklist.csv that the reviewer
fills in (one row per image: keep / lane_line_fp / other_fp / unsure).
"""

import argparse
import csv
import os
import random
import shutil


def list_label_files(label_dir):
    return sorted(f for f in os.listdir(label_dir) if f.endswith(".txt"))


def file_has_class(label_path, target_class):
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cls = int(line.split()[0])
            except (ValueError, IndexError):
                continue
            if cls == target_class:
                return True
    return False


def image_path_for_label(images_dir, label_fname):
    stem = os.path.splitext(label_fname)[0]
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = os.path.join(images_dir, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--total", type=int, default=50)
    ap.add_argument("--target-class", type=int, default=0, help="Class to oversample for targeted review (default 0 = Longitudinal Crack, the known lane-line risk class)")
    ap.add_argument("--target-fraction", type=float, default=0.6, help="Fraction of the sample that should contain the target class")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    images_dir = os.path.expanduser(args.images)
    labels_dir = os.path.expanduser(args.labels)
    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)

    random.seed(args.seed)

    all_labels = list_label_files(labels_dir)
    if not all_labels:
        raise SystemExit(f"No label files found in {labels_dir}")

    target_files = [f for f in all_labels if file_has_class(os.path.join(labels_dir, f), args.target_class)]
    other_files = [f for f in all_labels if f not in set(target_files)]

    n_target = min(int(round(args.total * args.target_fraction)), len(target_files))
    n_other = min(args.total - n_target, len(other_files))

    sampled = random.sample(target_files, n_target) + random.sample(other_files, n_other)
    random.shuffle(sampled)

    rows = []
    copied = 0
    for label_fname in sampled:
        img_path = image_path_for_label(images_dir, label_fname)
        if img_path is None:
            continue
        dest_name = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(out_dir, dest_name))
        copied += 1
        rows.append({
            "image_filename": dest_name,
            "label_filename": label_fname,
            "contains_target_class": label_fname in target_files,
            "review_result": "",  # reviewer fills in: keep / lane_line_fp / other_fp / unsure
            "notes": "",
        })

    csv_path = os.path.join(out_dir, "review_checklist.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_filename", "label_filename", "contains_target_class", "review_result", "notes"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Sampled {copied} images ({n_target} containing target class {args.target_class}, {n_other} other) -> {out_dir}")
    print(f"Checklist written to {csv_path}")
    print("\nFill in 'review_result' for each row with one of: keep / lane_line_fp / other_fp / unsure")
    print(f"Then re-run with the checklist to compute the false-positive rate (see compute_fp_rate.py).")


if __name__ == "__main__":
    main()
