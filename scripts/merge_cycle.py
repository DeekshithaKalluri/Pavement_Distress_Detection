"""
merge_cycle.py

Merges a new auto-labeled cycle (e.g. auto_labels/vid1_cycle1/) into a new
combined dataset directory alongside the existing baseline (full_remapped/),
without modifying either source in place. The combined output is what gets
re-split (80/10/10) and used for retraining.

Filename collisions between the baseline and the new cycle are checked
explicitly before copying anything -- given the baseline uses source-dataset
prefixes (e.g. "1.SVRDD_YOLO_...") and the auto-label cycle uses dashcam
video-name prefixes (e.g. "GRMN0015_f00000"), no collisions are expected,
but this is verified rather than assumed.

Usage:
    python merge_cycle.py \
        --baseline ~/Pavement_Distress_Detection/Dataset/full_remapped \
        --cycle ~/Pavement_Distress_Detection/auto_labels/vid1_cycle1 \
        --out ~/Pavement_Distress_Detection/Dataset/full_cycle1 \
        --mode symlink

--mode symlink (default) symlinks every file into the new combined directory,
matching the disk-space-conscious approach already used for Dataset/split/.
--mode copy makes real copies instead, if you'd rather the combined directory
be self-contained and independent of the two sources.
"""

import argparse
import os
import shutil


def list_pairs(images_dir, labels_dir):
    """Return {stem: (image_path, label_path)} for every image that has a
    matching label file. Images without a matching label are skipped with
    a warning, since that would indicate a structural problem worth knowing
    about rather than silently dropping."""
    pairs = {}
    skipped = []
    for fname in os.listdir(images_dir):
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        label_path = os.path.join(labels_dir, stem + ".txt")
        if os.path.exists(label_path):
            pairs[stem] = (os.path.join(images_dir, fname), label_path)
        else:
            skipped.append(fname)
    return pairs, skipped


def link_or_copy(src, dst, mode):
    if os.path.exists(dst):
        return  # idempotent: don't re-link/copy if already present
    if mode == "symlink":
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--cycle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    args = ap.parse_args()

    baseline_dir = os.path.expanduser(args.baseline)
    cycle_dir = os.path.expanduser(args.cycle)
    out_dir = os.path.expanduser(args.out)

    baseline_images = os.path.join(baseline_dir, "images")
    baseline_labels = os.path.join(baseline_dir, "labels")
    cycle_images = os.path.join(cycle_dir, "images")
    cycle_labels = os.path.join(cycle_dir, "labels")

    for p in (baseline_images, baseline_labels, cycle_images, cycle_labels):
        if not os.path.isdir(p):
            raise SystemExit(f"Expected directory not found: {p}")

    print(f"Scanning baseline: {baseline_dir}")
    baseline_pairs, baseline_skipped = list_pairs(baseline_images, baseline_labels)
    print(f"  {len(baseline_pairs)} image/label pairs, {len(baseline_skipped)} images skipped (no matching label)")

    print(f"Scanning new cycle: {cycle_dir}")
    cycle_pairs, cycle_skipped = list_pairs(cycle_images, cycle_labels)
    print(f"  {len(cycle_pairs)} image/label pairs, {len(cycle_skipped)} images skipped (no matching label)")

    # Explicit collision check -- do not proceed silently if any filename stem
    # appears in both sources, since that could mean one set silently
    # overwrites or shadows the other.
    collisions = set(baseline_pairs.keys()) & set(cycle_pairs.keys())
    if collisions:
        print(f"\nERROR: {len(collisions)} filename collision(s) between baseline and new cycle:")
        for c in sorted(collisions)[:10]:
            print(f"  - {c}")
        if len(collisions) > 10:
            print(f"  ... and {len(collisions) - 10} more")
        raise SystemExit(
            "Refusing to merge with unresolved filename collisions. "
            "Rename conflicting files or investigate why the same stem "
            "appears in both sources before retrying."
        )
    print("No filename collisions between baseline and new cycle. Safe to merge.")

    out_images = os.path.join(out_dir, "images")
    out_labels = os.path.join(out_dir, "labels")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    total_pairs = {**baseline_pairs, **cycle_pairs}
    print(f"\nMerging {len(total_pairs)} total image/label pairs into {out_dir} (mode={args.mode}) ...")

    for stem, (img_src, lbl_src) in total_pairs.items():
        img_ext = os.path.splitext(img_src)[1]
        link_or_copy(img_src, os.path.join(out_images, stem + img_ext), args.mode)
        link_or_copy(lbl_src, os.path.join(out_labels, stem + ".txt"), args.mode)

    # Carry over classes.txt from the baseline if present, since the combined
    # dataset uses the same 6-class scheme.
    baseline_classes = os.path.join(baseline_dir, "classes.txt")
    if os.path.exists(baseline_classes):
        shutil.copy2(baseline_classes, os.path.join(out_dir, "classes.txt"))
        print("Copied classes.txt from baseline.")

    print(f"\nDone. Combined dataset: {len(total_pairs)} pairs")
    print(f"  baseline contribution: {len(baseline_pairs)}")
    print(f"  new cycle contribution: {len(cycle_pairs)}")
    print(f"Output: {out_dir}")
    print("\nNext step: re-run the 80/10/10 split script against this new")
    print("combined directory (pointing it at this --out path instead of")
    print("the original full_remapped/), producing a fresh split for retraining.")


if __name__ == "__main__":
    main()
