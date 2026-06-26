"""
check_raw_vs_filtered.py

Compares class distribution BEFORE persistence filtering (raw per-frame
tracker output in the *_tracktmp/track_*/labels folders) against AFTER
filtering (the final auto_labels/<cycle>/labels folder).

This distinguishes two very different situations that look identical in
the final filtered output alone:
  1. A class was barely detected by the model at all in the raw tracker
     output -> consistent with that distress type simply not being present
     much in this video footage.
  2. A class WAS detected reasonably often in the raw output, but got
     dropped during persistence filtering (e.g. detections were too
     short-lived / too gappy to form a qualifying contiguous run) -> points
     to a tracking/filtering weakness rather than an absence in the footage.

Usage:
    python check_raw_vs_filtered.py \
        --raw-root ~/Pavement_Distress_Detection/auto_labels/vid1_cycle1_tracktmp \
        --filtered ~/Pavement_Distress_Detection/auto_labels/vid1_cycle1/labels
"""

import argparse
import os
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


def find_all_label_files(root, pattern_suffix="labels"):
    """Walk a directory tree and collect every .txt file under any folder
    ending in `pattern_suffix` (handles the per-video track_* subfolders)."""
    paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == pattern_suffix:
            for fname in filenames:
                if fname.endswith(".txt"):
                    paths.append(os.path.join(dirpath, fname))
    return paths


def count_classes_from_paths(paths, max_workers=64):
    counts = Counter()
    if not paths:
        return counts
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_count_one_file, p) for p in paths]
        for fut in as_completed(futures):
            counts.update(fut.result())
    return counts


def pct_table(counts):
    total = sum(counts.values())
    return {c: (counts.get(c, 0) / total * 100 if total else 0.0) for c in CLASS_NAMES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True, help="The *_tracktmp folder containing per-video track_*/labels subfolders")
    ap.add_argument("--filtered", required=True, help="The final filtered labels folder (post persistence-filtering)")
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    raw_root = os.path.expanduser(args.raw_root)
    filtered_dir = os.path.expanduser(args.filtered)

    print(f"Scanning raw tracker output under {raw_root} ...", flush=True)
    raw_paths = find_all_label_files(raw_root)
    print(f"Found {len(raw_paths)} raw per-frame label files. Counting classes...", flush=True)
    raw_counts = count_classes_from_paths(raw_paths, max_workers=args.workers)

    print(f"Counting classes in filtered output {filtered_dir} ...", flush=True)
    filtered_paths = [os.path.join(filtered_dir, f) for f in os.listdir(filtered_dir) if f.endswith(".txt")]
    filtered_counts = count_classes_from_paths(filtered_paths, max_workers=args.workers)

    raw_pct = pct_table(raw_counts)
    filt_pct = pct_table(filtered_counts)
    raw_total = sum(raw_counts.values())
    filt_total = sum(filtered_counts.values())

    print("=" * 92)
    print(f"Raw (pre-filter) detections:   {raw_total}")
    print(f"Filtered (kept) detections:    {filt_total}")
    print("=" * 92)
    print(f"{'Class':<22}{'Raw count':>12}{'Raw %':>10}{'Kept count':>12}{'Kept %':>10}{'Survival %':>12}")
    print("-" * 92)
    for cls, name in CLASS_NAMES.items():
        raw_n = raw_counts.get(cls, 0)
        kept_n = filtered_counts.get(cls, 0)
        survival = (kept_n / raw_n * 100) if raw_n > 0 else (0.0 if kept_n == 0 else float("inf"))
        survival_str = f"{survival:.1f}%" if survival != float("inf") else "n/a"
        flag = ""
        if raw_n == 0:
            flag = "  <-- never proposed by model"
        elif survival < 5.0:
            flag = "  <-- proposed but almost all filtered out"
        print(f"{name:<22}{raw_n:>12}{raw_pct[cls]:>9.2f}%{kept_n:>12}{filt_pct[cls]:>9.2f}%{survival_str:>12}{flag}")
    print("-" * 92)
    print("\nInterpretation guide:")
    print("  - 'never proposed by model' (raw count = 0): consistent with that distress type")
    print("    simply not appearing in this video footage -- not a pipeline problem.")
    print("  - 'proposed but almost all filtered out' (low survival %% despite raw count > 0):")
    print("    the model DID detect this class, but persistence filtering dropped nearly all")
    print("    of it -- worth checking MIN_TRACK_LENGTH / MAX_GAP for this class specifically,")
    print("    since it suggests a tracking-stability weakness rather than true absence.")
    print("=" * 92)


if __name__ == "__main__":
    main()
