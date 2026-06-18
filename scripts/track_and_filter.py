import shutil
import time
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO


def log(msg):
    print(msg, flush=True)


BAR_WIDTH = 20
BAR_CHECKPOINTS = 10


def make_bar(fraction):
    filled = int(round(fraction * BAR_WIDTH))
    filled = max(0, min(BAR_WIDTH, filled))
    return "[" + "#" * filled + "-" * (BAR_WIDTH - filled) + "]"

BASE = Path.home() / "Pavement_Distress_Detection"
FRAMES_DIR = BASE / "Dataset" / "frames" / "vid1"
MODEL_PATH = BASE / "runs" / "train" / "yolov11x" / "x.baseline" / "weights" / "best.pt"
OUTPUT_DIR = BASE / "auto_labels" / "vid1_cycle1"
TRACK_TMP_DIR = BASE / "auto_labels" / "vid1_cycle1_tracktmp"

MIN_TRACK_LENGTH = 3
CONF_FLOOR = 0.25
MAX_GAP = 1
SAMPLES_PER_RUN = 3

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "images").mkdir(exist_ok=True)
(OUTPUT_DIR / "labels").mkdir(exist_ok=True)

video_groups = defaultdict(list)
for img_path in sorted(FRAMES_DIR.glob("*.jpg")):
    stem = img_path.stem
    video_stem = stem.rsplit("_f", 1)[0]
    video_groups[video_stem].append(img_path)

log(f"Found {len(video_groups)} source videos in {FRAMES_DIR}")
log("Loading model...")
model = YOLO(str(MODEL_PATH))
log("Model loaded.")

total_kept_images = 0
total_dropped_images = 0
num_videos = len(video_groups)
overall_start = time.time()

for video_num, (video_stem, frame_paths) in enumerate(sorted(video_groups.items()), start=1):
    frame_paths = sorted(frame_paths)
    video_start = time.time()
    elapsed_so_far = video_start - overall_start
    log(f"\n[{video_num}/{num_videos}] Tracking video: {video_stem} ({len(frame_paths)} frames) -- elapsed so far: {elapsed_so_far/60:.1f} min")

    run_name = f"track_{video_stem}"
    shutil.rmtree(TRACK_TMP_DIR / run_name, ignore_errors=True)

    video_frame_dir = TRACK_TMP_DIR / f"{video_stem}_frames"
    shutil.rmtree(video_frame_dir, ignore_errors=True)
    video_frame_dir.mkdir(parents=True, exist_ok=True)
    for frame_path in frame_paths:
        (video_frame_dir / frame_path.name).symlink_to(frame_path)
    log(f"  Symlinked {len(frame_paths)} frames, starting tracking...")

    num_frames_this_video = len(frame_paths)
    progress_state = {"frame_count": 0, "next_checkpoint": 1, "detections_found": 0}

    def on_predict_batch_end(predictor):
        try:
            progress_state["frame_count"] += 1
            frame_count = progress_state["frame_count"]

            try:
                if predictor.results and len(predictor.results[-1].boxes) > 0:
                    progress_state["detections_found"] += len(predictor.results[-1].boxes)
            except (AttributeError, IndexError, TypeError):
                pass

            if num_frames_this_video > 0:
                progress_fraction = frame_count / num_frames_this_video
                checkpoint_fraction = progress_state["next_checkpoint"] / BAR_CHECKPOINTS
                if progress_fraction >= checkpoint_fraction and progress_state["next_checkpoint"] <= BAR_CHECKPOINTS:
                    pct = int(round(progress_fraction * 100))
                    bar = make_bar(progress_fraction)
                    log(
                        f"  [{video_num}/{num_videos}] {video_stem}: {bar} {pct}% "
                        f"({frame_count}/{num_frames_this_video} frames, "
                        f"{progress_state['detections_found']} detections so far)"
                    )
                    progress_state["next_checkpoint"] += 1
        except Exception:
            pass

    callback_registered = False
    try:
        model.add_callback("on_predict_batch_end", on_predict_batch_end)
        callback_registered = True
    except Exception as e:
        log(f"  WARNING: could not register progress callback ({e}); falling back to default verbose output")

    model.track(
        source=str(video_frame_dir),
        imgsz=896,
        conf=CONF_FLOOR,
        iou=0.45,
        tracker="bytetrack.yaml",
        save=False,
        save_txt=True,
        save_conf=True,
        project=str(TRACK_TMP_DIR),
        name=run_name,
        exist_ok=True,
        device=0,
        persist=False,
        verbose=not callback_registered,
    )

    if callback_registered:
        try:
            model.clear_callback("on_predict_batch_end")
        except Exception:
            pass

    label_dir = TRACK_TMP_DIR / run_name / "labels"
    if not label_dir.exists():
        log(f"  No labels directory produced for {video_stem}, skipping")
        shutil.rmtree(video_frame_dir, ignore_errors=True)
        continue

    frame_lines = {}
    for idx, frame_path in enumerate(frame_paths):
        label_path = label_dir / f"{frame_path.stem}.txt"
        lines = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 7:
                        continue
                    lines.append(parts)
        frame_lines[idx] = lines

    track_appearances = defaultdict(list)
    for idx in range(len(frame_paths)):
        for parts in frame_lines[idx]:
            track_appearances[parts[6]].append(idx)

    run_length_by_frame_and_track = {}
    frames_to_keep_for_track_run = set()

    for track_id, frame_indices in track_appearances.items():
        frame_indices = sorted(frame_indices)
        run_start = 0
        for i in range(1, len(frame_indices) + 1):
            at_end = i == len(frame_indices)
            gap_too_large = (not at_end) and (frame_indices[i] - frame_indices[i - 1] > MAX_GAP + 1)
            if at_end or gap_too_large:
                run = frame_indices[run_start:i]
                run_len = len(run)
                for fidx in run:
                    run_length_by_frame_and_track[(fidx, track_id)] = run_len

                if run_len >= MIN_TRACK_LENGTH:
                    if run_len <= SAMPLES_PER_RUN:
                        sampled = run
                    else:
                        step_positions = [
                            round(j * (run_len - 1) / (SAMPLES_PER_RUN - 1))
                            for j in range(SAMPLES_PER_RUN)
                        ]
                        sampled = sorted(set(run[pos] for pos in step_positions))
                    for fidx in sampled:
                        frames_to_keep_for_track_run.add((fidx, track_id))

                run_start = i

    video_kept = 0
    video_dropped = 0
    for idx, frame_path in enumerate(frame_paths):
        kept_lines = [
            f"{p[0]} {p[1]} {p[2]} {p[3]} {p[4]}"
            for p in frame_lines[idx]
            if (idx, p[6]) in frames_to_keep_for_track_run
        ]

        if kept_lines:
            out_label = OUTPUT_DIR / "labels" / f"{frame_path.stem}.txt"
            with open(out_label, "w") as f:
                f.write("\n".join(kept_lines))
            shutil.copy(frame_path, OUTPUT_DIR / "images" / frame_path.name)
            video_kept += 1
        else:
            video_dropped += 1

    total_kept_images += video_kept
    total_dropped_images += video_dropped
    video_elapsed = time.time() - video_start
    log(
        f"  [{video_num}/{num_videos}] Done: kept {video_kept}, dropped {video_dropped} "
        f"({video_elapsed:.1f}s) -- running totals: kept {total_kept_images}, dropped {total_dropped_images}"
    )

    shutil.rmtree(TRACK_TMP_DIR / run_name, ignore_errors=True)
    shutil.rmtree(video_frame_dir, ignore_errors=True)

shutil.rmtree(TRACK_TMP_DIR, ignore_errors=True)

overall_elapsed = time.time() - overall_start
log(f"\n{'='*50}")
log(f"Kept   : {total_kept_images} image/label pairs (contiguous track run >= {MIN_TRACK_LENGTH} frames, gap tolerance = {MAX_GAP})")
log(f"Dropped: {total_dropped_images} frames (no surviving runs)")
log(f"Total time: {overall_elapsed/60:.1f} min")
log(f"Output : {OUTPUT_DIR}")