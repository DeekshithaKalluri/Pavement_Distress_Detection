"""
raw_tracking_sample.py

Runs YOLOv11x in tracking mode (no persistence filtering) on a small,
deliberately-chosen sample of videos, and reports the raw per-class
detection counts. Intended as a quick, cheap diagnostic to answer one
question: does the model ever propose Block Crack / Patch / Alligator
Crack at all in this footage, before any filtering happens?

This intentionally does NOT delete its temp output, so the raw labels
remain available for inspection afterward (unlike the full pipeline run,
which cleans up per-video temp folders once filtering is complete).

Usage:
    python raw_tracking_sample.py \
        --weights ~/Pavement_Distress_Detection/runs/train/yolov11x/x.baseline/weights/best.pt \
        --frames-root ~/Pavement_Distress_Detection/frames/vid1 \
        --out ~/Pavement_Distress_Detection/auto_labels/raw_sample_check \
        --num-videos 5 \
        --conf 0.25

`--frames-root` should point at the same extracted-frames directory the
main pipeline uses (one subfolder per source video, e.g. vid1_GRMN0015/).
By default this picks --num-videos videos spread evenly across the full
list (not just the first few), so the sample isn't biased toward whatever
happened to be recorded first.
"""

import argparse
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict

CLASS_NAMES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Pothole",
    4: "Patch",
    5: "Block Crack",
}

# Matches frame filenames like "GRMN0015_f00003.jpg" -> video name "GRMN0015"
FRAME_PATTERN = re.compile(r"^(.+)_f\d+\.(jpg|jpeg|png)$", re.IGNORECASE)


def group_frames_by_video(frames_root):
    """Frames are stored flat (not one-subfolder-per-video): every video's
    frames live directly in frames_root, named <VIDEONAME>_f<FRAMENUM>.jpg.
    Group them back into per-video lists by parsing the filename prefix."""
    groups = defaultdict(list)
    for fname in os.listdir(frames_root):
        m = FRAME_PATTERN.match(fname)
        if not m:
            continue
        video_name = m.group(1)
        groups[video_name].append(fname)
    for video_name in groups:
        groups[video_name].sort()
    return groups


def pick_spread_sample(video_names, n):
    video_names = sorted(video_names)
    if n >= len(video_names):
        return video_names
    step = len(video_names) / n
    return [video_names[int(i * step)] for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--frames-root", required=True, help="Flat directory containing all extracted frames, named <VIDEONAME>_f<FRAMENUM>.jpg")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-videos", type=int, default=5)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    from ultralytics import YOLO  # imported here so --help works without the env loaded

    frames_root = os.path.expanduser(args.frames_root)
    out_root = os.path.expanduser(args.out)
    os.makedirs(out_root, exist_ok=True)

    print(f"Scanning {frames_root} for frame files and grouping by source video...", flush=True)
    video_groups = group_frames_by_video(frames_root)
    if not video_groups:
        raise SystemExit(
            f"No frame files matching '<VIDEONAME>_f<FRAMENUM>.jpg' found under {frames_root}. "
            f"Check that --frames-root points at the flat frames directory."
        )
    print(f"Found {len(video_groups)} distinct videos, {sum(len(v) for v in video_groups.values())} frames total.")

    sample = pick_spread_sample(list(video_groups.keys()), args.num_videos)
    print(f"Sampling {len(sample)} of {len(video_groups)} videos: {sample}")

    model = YOLO(os.path.expanduser(args.weights))

    overall_counts = Counter()

    with tempfile.TemporaryDirectory(prefix="raw_track_sample_") as tmp_base:
        for video_name in sample:
            frame_files = video_groups[video_name]
            video_tmp_dir = os.path.join(tmp_base, video_name)
            os.makedirs(video_tmp_dir, exist_ok=True)
            for fname in frame_files:
                src = os.path.join(frames_root, fname)
                dst = os.path.join(video_tmp_dir, fname)
                if not os.path.exists(dst):
                    os.symlink(src, dst)

            print(f"\nTracking (raw, no filtering): {video_name} ({len(frame_files)} frames)")
            model.track(
                source=video_tmp_dir,
                conf=args.conf,
                save=False,
                save_txt=True,
                save_conf=True,
                project=out_root,
                name=video_name,
                persist=True,
                verbose=False,
            )

            label_dir = os.path.join(out_root, video_name, "labels")
            video_counts = Counter()
            if os.path.isdir(label_dir):
                for fname in os.listdir(label_dir):
                    if not fname.endswith(".txt"):
                        continue
                    with open(os.path.join(label_dir, fname)) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                cls = int(line.split()[0])
                            except (ValueError, IndexError):
                                continue
                            video_counts[cls] += 1
            overall_counts.update(video_counts)
            print(f"  {video_name}: {dict(video_counts)}")

    print("\n" + "=" * 70)
    print(f"RAW (unfiltered) detection counts across {len(sample)} sampled videos:")
    print("-" * 70)
    total = sum(overall_counts.values())
    for cls, name in CLASS_NAMES.items():
        n = overall_counts.get(cls, 0)
        pct = (n / total * 100) if total else 0.0
        flag = "  <-- zero raw detections even before filtering" if n == 0 else ""
        print(f"  {name:<22}{n:>8}  ({pct:5.1f}%){flag}")
    print("=" * 70)
    print("\nRaw label files preserved under:", out_root)
    print("(not deleted, unlike the main pipeline's temp folders)")


if __name__ == "__main__":
    main()
