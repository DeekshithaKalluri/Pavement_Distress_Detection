"""
check_class_distribution.py

Compares the class distribution of newly auto-labeled data against the
baseline training set distribution. Flags any class whose share has
shifted dramatically, which could indicate a systematic labeling bias
(e.g., lane lines being mass-labeled as longitudinal cracks) before the
new labels get merged into the training set.

Usage:
    python check_class_distribution.py \
        --baseline ~/Pavement_Distress_Detection/Dataset/split/train/labels \
        --new ~/Pavement_Distress_Detection/auto_labels/vid1_cycle1/labels \
        --flag-ratio 2.0

A class is flagged if its share of the new batch is more than `--flag-ratio`
times higher OR lower than its share in the baseline (only for classes that
appear meaningfully often in the baseline; very rare baseline classes use an
absolute floor instead of a ratio, since tiny baseline counts make ratios
unstable).
"""

import argparse
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

CLASS_NAMES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Pothole",
    4: "Patch",
    5: "Block Crack",
}


def _count_one_file(path):
    """Read a single label file and return a Counter of class occurrences.
    Kept as a standalone function so it can be dispatched to a thread pool --
    each call is I/O-bound (one open/read/close), so threads (not processes)
    give a large speedup on network-filesystem-backed directories with many
    small files, since most of the wall time is spent waiting on I/O, not
    executing Python bytecode."""
    local = Counter()
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    cls = int(line.split()[0])
                except (ValueError, IndexError):
                    continue
                local[cls] += 1
    except OSError:
        pass
    return local


def count_classes(label_dir, max_workers=64, progress_label=""):
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    fnames = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    total_files = len(fnames)
    counts = Counter()

    if total_files == 0:
        return counts, 0

    paths = [os.path.join(label_dir, f) for f in fnames]
    done = 0
    last_print = time.time()
    start = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_count_one_file, p) for p in paths]
        for fut in as_completed(futures):
            counts.update(fut.result())
            done += 1
            now = time.time()
            if progress_label and (now - last_print > 5 or done == total_files):
                elapsed = now - start
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  [{progress_label}] {done}/{total_files} files "
                      f"({rate:.0f} files/sec, {elapsed:.0f}s elapsed)",
                      flush=True)
                last_print = now

    return counts, total_files


def pct_table(counts):
    total = sum(counts.values())
    return {c: (counts.get(c, 0) / total * 100 if total else 0.0) for c in CLASS_NAMES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="Path to baseline label directory (e.g. Dataset/split/train/labels)")
    ap.add_argument("--new", required=True, help="Path to new auto-labeled label directory")
    ap.add_argument("--flag-ratio", type=float, default=2.0, help="Flag if new share is more than this many times higher/lower than baseline share")
    ap.add_argument("--min-baseline-pct", type=float, default=1.0, help="Below this baseline %% share, use absolute floor instead of ratio (avoids unstable ratios for rare classes)")
    ap.add_argument("--workers", type=int, default=64, help="Concurrent file-read threads (higher helps on slow network filesystems; lower if it overloads the system)")
    args = ap.parse_args()

    print("Reading baseline labels (this can take a little while on a large training set)...", flush=True)
    baseline_counts, baseline_files = count_classes(os.path.expanduser(args.baseline), max_workers=args.workers, progress_label="baseline")
    print("Reading new batch labels...", flush=True)
    new_counts, new_files = count_classes(os.path.expanduser(args.new), max_workers=args.workers, progress_label="new")

    baseline_total = sum(baseline_counts.values())
    new_total = sum(new_counts.values())

    baseline_pct = pct_table(baseline_counts)
    new_pct = pct_table(new_counts)

    print("=" * 78)
    print(f"Baseline: {args.baseline}")
    print(f"  {baseline_files} label files, {baseline_total} annotations")
    print(f"New batch: {args.new}")
    print(f"  {new_files} label files, {new_total} annotations")
    print("=" * 78)
    print(f"{'Class':<22}{'Baseline %':>12}{'New %':>10}{'Ratio':>10}  Flag")
    print("-" * 78)

    flags = []
    for cls, name in CLASS_NAMES.items():
        b_pct = baseline_pct[cls]
        n_pct = new_pct[cls]
        if b_pct >= args.min_baseline_pct:
            ratio = (n_pct / b_pct) if b_pct > 0 else float("inf")
            flagged = ratio > args.flag_ratio or ratio < (1.0 / args.flag_ratio)
        else:
            # Rare baseline class: use an absolute-point-difference floor instead.
            flagged = abs(n_pct - b_pct) > (args.min_baseline_pct * args.flag_ratio)
            ratio = (n_pct / b_pct) if b_pct > 0 else float("inf")

        ratio_str = f"{ratio:.2f}x" if ratio != float("inf") else "inf"
        flag_str = "  <-- FLAG" if flagged else ""
        print(f"{name:<22}{b_pct:>11.2f}%{n_pct:>9.2f}%{ratio_str:>10}{flag_str}")
        if flagged:
            flags.append((name, b_pct, n_pct))

    print("-" * 78)
    if flags:
        print(f"\n{len(flags)} class(es) flagged for review before merging:")
        for name, b_pct, n_pct in flags:
            direction = "OVER-represented" if n_pct > b_pct else "UNDER-represented"
            print(f"  - {name}: baseline {b_pct:.2f}% -> new {n_pct:.2f}%  ({direction})")
        print("\nRecommendation: manually inspect a sample of the flagged class(es)")
        print("in the new batch before merging into the training set.")
    else:
        print("\nNo classes flagged. Distribution looks broadly consistent with baseline.")
    print("=" * 78)


if __name__ == "__main__":
    main()
