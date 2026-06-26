#!/usr/bin/env python3
"""
auto_label_pipeline.py

Single-command, end-to-end pavement distress auto-labeling pipeline.

Combines the two previously separate steps used in this project
(extract_frames.py + track_and_filter.py) into one script, and adds:

  - automatic input-type detection: if given videos, extracts frames first;
    if given a folder of images, skips straight to tracking/filtering
  - GPS lat/long extraction via OCR of the on-screen Garmin text overlay
  - a labeled output image (boxes + class + confidence drawn on) alongside
    the existing plain YOLO .txt label, for every kept detection
  - a bounding-box diagonal length in PIXELS for every kept detection, with
    an explicit real_world_length_m placeholder (currently null) reserved
    for a future camera-calibration step

WHAT THIS DOES NOT DO (read before relying on outputs):

  - GPS via OCR is NOT guaranteed correct. It reads the on-screen text
    burned into the video by the dashcam, the same way a human reading the
    corner of the screen would, and can misread digits due to glare, motion
    blur, or compression artifacts. Every detection's GPS field is reported
    together with an `gps_ocr_confidence` value and is `null` if OCR could
    not parse a plausible-looking coordinate, rather than guessing. Treat
    OCR'd coordinates as approximate location context, not survey-grade GPS.

  - `bbox_diagonal_px` is a PIXEL measurement of the box diagonal, not a
    real-world distance. A box's pixel size depends on how far the object
    is from the camera and the camera's mounting angle/height/field of
    view -- none of which this script knows. Converting this to a real
    physical length (e.g. meters) requires a camera calibration step
    (known mounting height, tilt angle, focal length / field of view) that
    has not been provided. `real_world_length_m` is therefore always
    written as null for now; the diagonal-pixel value and the underlying
    box geometry are preserved so that calibration can be applied
    retroactively later without re-running detection.

Usage:
    python auto_label_pipeline.py --input /path/to/video_or_folder

MODEL_PATH and OUTPUT_BASE_DIR are fixed constants near the top of this file
(edit them there if your paths differ) -- only --input varies per run.

--input may be:
  - a single video file (e.g. Dataset/GRMN0015.MP4)
  - a folder containing multiple video files
  - a folder containing pre-extracted images (frame extraction is skipped)

Each run creates a new auto-numbered output folder under OUTPUT_BASE_DIR,
named e.g. 01_GRMN0015.MP4_output, 02_GRMN0016.MP4_output, etc., so repeated
runs on different inputs never collide and stay easy to tell apart.

If --input contains video files (.mp4/.MP4/.mov/.avi/.mkv), frames are
extracted first (same logic as extract_frames.py). If --input contains
image files directly, frame extraction is skipped entirely and those
images are used as-is -- they must already be named so that frames from
the same source video are grouped correctly, i.e.
<BATCH_TAG>_<VIDEO_STEM>_f<NNNNN>.jpg, matching this project's existing
naming convention. If your images aren't named this way, group/rename them
before running, or treat them as a single one-frame "video" each (which
will work, but every persistence/tracking benefit is lost when there's
only one frame per group).
"""

import argparse
import csv
import json
import math
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Fixed configuration -- edit these constants directly rather than passing
# them on the command line, since the model and base output location don't
# change between runs in normal use. Only --input varies per run.
# ---------------------------------------------------------------------------

MODEL_PATH = Path.home() / "Pavement_Distress_Detection" / "runs" / "train" / "yolov11x" / "x.baseline" / "weights" / "best.pt"
OUTPUT_BASE_DIR = Path.home() / "Pavement_Distress_Detection" / "auto_labels"

FPS_EXTRACT = 4
MIN_TRACK_LENGTH = 3
MAX_GAP = 1
CONF_FLOOR = 0.25
SAMPLES_PER_RUN = 3
RUN_GPS_OCR = True

# ---------------------------------------------------------------------------
# Shared progress-bar helper (same style as the existing two scripts)
# ---------------------------------------------------------------------------

BAR_WIDTH = 20
BAR_CHECKPOINTS = 10


def make_bar(fraction):
    filled = int(round(fraction * BAR_WIDTH))
    filled = max(0, min(BAR_WIDTH, filled))
    return "[" + "#" * filled + "-" * (BAR_WIDTH - filled) + "]"


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Stage 0: input-type detection
# ---------------------------------------------------------------------------

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def detect_input_type(input_path):
    """Returns ('video', [list of video file Paths]) or ('image', input_dir)
    based on what --input actually points at:
      - a single video file -> ('video', [that file])
      - a folder of videos   -> ('video', [all video files in it])
      - a folder of images   -> ('image', input_path)
    Mixed video+image folders are rejected rather than silently picking one,
    since silently ignoring half the input is a worse failure mode than
    stopping and asking the user to separate them."""
    if input_path.is_file():
        if input_path.suffix.lower() in VIDEO_EXTS:
            return "video", [input_path]
        raise SystemExit(f"{input_path} is a file but not a recognized video type ({sorted(VIDEO_EXTS)})")

    if not input_path.is_dir():
        raise SystemExit(f"--input path does not exist: {input_path}")

    files = list(input_path.iterdir())
    video_files = sorted(f for f in files if f.suffix.lower() in VIDEO_EXTS)
    has_image = any(f.suffix.lower() in IMAGE_EXTS for f in files)

    if video_files and has_image:
        raise SystemExit(
            f"{input_path} contains both video and image files. "
            f"Put videos and pre-extracted images in separate input folders "
            f"and run this script once per folder."
        )
    if video_files:
        return "video", video_files
    if has_image:
        return "image", input_path
    raise SystemExit(f"No video or image files found in {input_path}")


def next_numbered_output_dir(base_dir, label):
    """Builds an auto-incrementing output folder name like
    01_GRMN0015.MP4_output, 02_GRMN0016.MP4_output, etc. so that running
    this script repeatedly on different inputs produces clearly ordered,
    non-colliding output folders without the user having to track numbers
    themselves. The number is based on how many NN_*_output folders already
    exist directly under base_dir -- it does not try to detect or reuse
    gaps if folders were deleted, it always picks (highest existing N) + 1,
    so numbering only ever goes forward."""
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = [d.name for d in base_dir.iterdir() if d.is_dir()]
    highest = 0
    for name in existing:
        m = re.match(r"^(\d+)_", name)
        if m:
            highest = max(highest, int(m.group(1)))
    next_num = highest + 1
    safe_label = re.sub(r"[^A-Za-z0-9._-]", "_", label)
    return base_dir / f"{next_num:02d}_{safe_label}_output"


# ---------------------------------------------------------------------------
# Stage 1: frame extraction (skipped entirely if input is already images)
# ---------------------------------------------------------------------------

def extract_frames(video_files, frames_dir, batch_tag, fps_extract=4):
    frames_dir.mkdir(parents=True, exist_ok=True)

    log(f"[extract] Processing {len(video_files)} video(s)")

    total_frames_saved = 0
    for video_num, video_path in enumerate(video_files, start=1):
        log(f"[extract] [{video_num}/{len(video_files)}] Starting: {video_path.name}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            log(f"[extract]   WARNING: could not open {video_path.name}, skipping")
            continue

        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, round(native_fps / fps_extract))

        video_stem = video_path.stem
        frame_idx = 0
        saved_idx = 0
        next_checkpoint = 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                out_name = f"{batch_tag}_{video_stem}_f{saved_idx:05d}.jpg"
                cv2.imwrite(str(frames_dir / out_name), frame)
                saved_idx += 1
            frame_idx += 1

            if total_video_frames > 0:
                progress_fraction = frame_idx / total_video_frames
                checkpoint_fraction = next_checkpoint / BAR_CHECKPOINTS
                if progress_fraction >= checkpoint_fraction and next_checkpoint <= BAR_CHECKPOINTS:
                    pct = int(round(progress_fraction * 100))
                    bar = make_bar(progress_fraction)
                    log(f"[extract]   [{video_num}/{len(video_files)}] {video_path.name}: "
                        f"{bar} {pct}% ({frame_idx}/{total_video_frames} frames read, {saved_idx} saved)")
                    next_checkpoint += 1

        cap.release()
        total_frames_saved += saved_idx
        log(f"[extract] [{video_num}/{len(video_files)}] Done: saved {saved_idx} frames "
            f"(running total: {total_frames_saved})")

    log(f"[extract] Total frames extracted: {total_frames_saved} -> {frames_dir}")
    return frames_dir


# ---------------------------------------------------------------------------
# Stage 1b: GPS OCR on the on-screen Garmin text overlay
# ---------------------------------------------------------------------------

# Matches lines like: "GARMIN 10/21/2025 01:43:42 PM 39.18652 -96.58547 4 MPH"
# Captures the two decimal coordinates regardless of exact spacing. The
# longitude's minus sign is matched separately and tolerantly (one or more
# dashes) because OCR on the burned-in overlay sometimes reads the single
# minus sign as a doubled dash (e.g. "--96.58547") depending on font
# rendering and compression artifacts at the text's exact pixel boundary.
GPS_PATTERN = re.compile(r"(-?\d{1,3}\.\d{3,6})\s*-+\s*(\d{1,3}\.\d{3,6})|(-?\d{1,3}\.\d{3,6})\D+(-?\d{1,3}\.\d{3,6})")


def _parse_gps_match(match):
    """The pattern above has two alternative capture-group sets (one for the
    doubled-dash case, one for the general case) -- pick whichever matched."""
    if match.group(1) is not None:
        lat = float(match.group(1))
        lon = -float(match.group(2))  # the consumed "-+" means this is always negative
    else:
        lat = float(match.group(3))
        lon = float(match.group(4))
    return lat, lon


def ocr_gps_from_frame(image_bgr, strip_height_frac=0.06):
    """Crops the bottom text-overlay strip of the frame and OCRs it for a
    lat/long pair. Returns (lat, lon, raw_ocr_text, confidence) where lat/lon
    are None if no plausible coordinate pair was parsed -- this deliberately
    does not guess or fall back to a best-effort partial match, since a
    wrong-looking-right coordinate is worse than an honest null."""
    try:
        import pytesseract
    except ImportError:
        return None, None, None, 0.0

    h, w = image_bgr.shape[:2]
    strip_top = int(h * (1 - strip_height_frac))
    strip = image_bgr[strip_top:h, 0:w]

    # Upscale and binarize -- small burned-in text OCRs far more reliably
    # at higher resolution and with a simple threshold than at native size
    # on a busy road-scene background.
    strip = cv2.resize(strip, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    try:
        data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
    except Exception:
        return None, None, None, 0.0

    raw_text = " ".join(t for t in data.get("text", []) if t.strip())
    confs = [int(c) for c in data.get("conf", []) if str(c).strip() not in ("", "-1")]
    avg_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0

    match = GPS_PATTERN.search(raw_text)
    if not match:
        return None, None, raw_text, avg_conf

    try:
        lat, lon = _parse_gps_match(match)
    except ValueError:
        return None, None, raw_text, avg_conf

    # Sanity bounds -- reject obviously-impossible coordinates rather than
    # passing along an OCR misread (e.g. a stray digit producing lat=391.8).
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None, None, raw_text, avg_conf

    return lat, lon, raw_text, avg_conf


# ---------------------------------------------------------------------------
# Stage 2: tracking + persistence filtering (same logic as track_and_filter.py)
# ---------------------------------------------------------------------------

def bbox_diagonal_px(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def draw_annotated_image(image_bgr, detections, class_names):
    """Draws boxes + "<class> <conf>" labels, same visual style used
    elsewhere in this project. Returns a new array; does not mutate input."""
    out = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        cls_id = det["class_id"]
        conf = det["confidence"]
        name = class_names.get(cls_id, str(cls_id))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{name} {conf:.2f}"
        cv2.putText(out, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def track_and_filter(frames_dir, model_path, output_dir, batch_tag,
                      min_track_length=3, conf_floor=0.25, max_gap=1,
                      samples_per_run=3, run_ocr=True):
    from ultralytics import YOLO

    track_tmp_dir = output_dir / f"{batch_tag}_tracktmp"
    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    annotated_out = output_dir / "images_annotated"
    for d in (images_out, labels_out, annotated_out):
        d.mkdir(parents=True, exist_ok=True)

    video_groups = defaultdict(list)
    for img_path in sorted(frames_dir.glob("*.jpg")):
        stem = img_path.stem
        # Filenames are "<batch_tag>_<video_stem>_f<NNNNN>.jpg". Strip the
        # batch_tag prefix first (if present) before splitting off "_f...",
        # so the recovered video_stem doesn't end up containing the batch
        # tag a second time when video_stem and batch_tag happen to be the
        # same string (e.g. a single-video run named after that video).
        without_tag = stem[len(batch_tag) + 1:] if stem.startswith(batch_tag + "_") else stem
        video_stem = without_tag.rsplit("_f", 1)[0]
        video_groups[video_stem].append(img_path)

    log(f"[track] Found {len(video_groups)} source videos/groups in {frames_dir}")
    log("[track] Loading model...")
    model = YOLO(str(model_path))
    class_names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    log("[track] Model loaded.")

    total_kept_images = 0
    total_dropped_images = 0
    total_kept_detections = 0
    gps_ocr_attempts = 0
    gps_ocr_successes = 0
    num_videos = len(video_groups)
    overall_start = time.time()

    manifest = []  # one entry per kept image, written to manifest.json at the end

    for video_num, (video_stem, frame_paths) in enumerate(sorted(video_groups.items()), start=1):
        frame_paths = sorted(frame_paths)
        video_start = time.time()
        elapsed_so_far = video_start - overall_start
        log(f"\n[track] [{video_num}/{num_videos}] Tracking video: {video_stem} "
            f"({len(frame_paths)} frames) -- elapsed so far: {elapsed_so_far/60:.1f} min")

        run_name = f"track_{video_stem}"
        shutil.rmtree(track_tmp_dir / run_name, ignore_errors=True)
        video_frame_dir = track_tmp_dir / f"{video_stem}_frames"
        shutil.rmtree(video_frame_dir, ignore_errors=True)
        video_frame_dir.mkdir(parents=True, exist_ok=True)
        for frame_path in frame_paths:
            (video_frame_dir / frame_path.name).symlink_to(frame_path)
        log(f"[track]   Symlinked {len(frame_paths)} frames, starting tracking...")

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
                        log(f"[track]   [{video_num}/{num_videos}] {video_stem}: {bar} {pct}% "
                            f"({frame_count}/{num_frames_this_video} frames, "
                            f"{progress_state['detections_found']} detections so far)")
                        progress_state["next_checkpoint"] += 1
            except Exception:
                pass

        callback_registered = False
        try:
            model.add_callback("on_predict_batch_end", on_predict_batch_end)
            callback_registered = True
        except Exception as e:
            log(f"[track]   WARNING: could not register progress callback ({e}); using default verbose output")

        model.track(
            source=str(video_frame_dir),
            imgsz=896,
            conf=conf_floor,
            iou=0.45,
            tracker="bytetrack.yaml",
            save=False,
            save_txt=True,
            save_conf=True,
            project=str(track_tmp_dir),
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

        label_dir = track_tmp_dir / run_name / "labels"
        if not label_dir.exists():
            log(f"[track]   No labels directory produced for {video_stem}, skipping")
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

        frames_to_keep_for_track_run = set()
        for track_id, frame_indices in track_appearances.items():
            frame_indices = sorted(frame_indices)
            run_start = 0
            for i in range(1, len(frame_indices) + 1):
                at_end = i == len(frame_indices)
                gap_too_large = (not at_end) and (frame_indices[i] - frame_indices[i - 1] > max_gap + 1)
                if at_end or gap_too_large:
                    run = frame_indices[run_start:i]
                    run_len = len(run)
                    if run_len >= min_track_length:
                        if run_len <= samples_per_run:
                            sampled = run
                        else:
                            step_positions = [
                                round(j * (run_len - 1) / (samples_per_run - 1))
                                for j in range(samples_per_run)
                            ]
                            sampled = sorted(set(run[pos] for pos in step_positions))
                        for fidx in sampled:
                            frames_to_keep_for_track_run.add((fidx, track_id))
                    run_start = i

        video_kept = 0
        video_dropped = 0
        for idx, frame_path in enumerate(frame_paths):
            kept_parts = [
                p for p in frame_lines[idx]
                if (idx, p[6]) in frames_to_keep_for_track_run
            ]
            if not kept_parts:
                video_dropped += 1
                continue

            image_bgr = cv2.imread(str(frame_path))
            img_h, img_w = image_bgr.shape[:2]

            lat, lon, gps_raw_text, gps_conf = (None, None, None, 0.0)
            if run_ocr:
                gps_ocr_attempts += 1
                lat, lon, gps_raw_text, gps_conf = ocr_gps_from_frame(image_bgr)
                if lat is not None:
                    gps_ocr_successes += 1

            detections_for_manifest = []
            label_lines = []
            for p in kept_parts:
                cls_id = int(p[0])
                xc, yc, bw, bh = float(p[1]), float(p[2]), float(p[3]), float(p[4])
                conf = float(p[5])
                x1 = int((xc - bw / 2) * img_w)
                y1 = int((yc - bh / 2) * img_h)
                x2 = int((xc + bw / 2) * img_w)
                y2 = int((yc + bh / 2) * img_h)
                diag_px = bbox_diagonal_px(x1, y1, x2, y2)

                label_lines.append(f"{cls_id} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]}")
                detections_for_manifest.append({
                    "class_id": cls_id,
                    "class_name": class_names.get(cls_id, str(cls_id)),
                    "confidence": conf,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "bbox_diagonal_px": round(diag_px, 1),
                    "real_world_length_m": None,  # placeholder -- requires camera calibration, not yet provided
                })
                total_kept_detections += 1

            out_label = labels_out / f"{frame_path.stem}.txt"
            with open(out_label, "w") as f:
                f.write("\n".join(label_lines))
            shutil.copy(frame_path, images_out / frame_path.name)

            annotated = draw_annotated_image(
                image_bgr,
                [{"x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"],
                  "class_id": d["class_id"], "confidence": d["confidence"]}
                 for d in detections_for_manifest],
                class_names,
            )
            cv2.imwrite(str(annotated_out / frame_path.name), annotated)

            manifest.append({
                "image": frame_path.name,
                "source_video": video_stem,
                "gps": {
                    "lat": lat,
                    "lon": lon,
                    "ocr_confidence": round(gps_conf, 3) if lat is not None else None,
                    "raw_ocr_text": gps_raw_text,
                } if run_ocr else None,
                "detections": detections_for_manifest,
            })

            video_kept += 1

        total_kept_images += video_kept
        total_dropped_images += video_dropped
        video_elapsed = time.time() - video_start
        log(f"[track]   [{video_num}/{num_videos}] Done: kept {video_kept}, dropped {video_dropped} "
            f"({video_elapsed:.1f}s) -- running totals: kept {total_kept_images}, dropped {total_dropped_images}")

        shutil.rmtree(track_tmp_dir / run_name, ignore_errors=True)
        shutil.rmtree(video_frame_dir, ignore_errors=True)

    shutil.rmtree(track_tmp_dir, ignore_errors=True)

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Simple, spreadsheet-friendly CSV alongside the full manifest.json --
    # one row per kept image, image_name + GPS only. This is intentionally
    # separate from the YOLO .txt label files (which must stay in plain
    # "class x y w h conf" format for training to work correctly) and is
    # also separate from manifest.json's full per-box detail, since GPS is
    # a per-image property, not a per-box one, and a CSV is the simplest
    # format for joining image -> location for mapping/repair-routing use
    # outside of this pipeline.
    metadata_csv_path = output_dir / "metadata.csv"
    with open(metadata_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "latitude", "longitude", "gps_ocr_confidence"])
        for entry in manifest:
            gps = entry.get("gps") or {}
            writer.writerow([
                entry["image"],
                gps.get("lat", ""),
                gps.get("lon", ""),
                gps.get("ocr_confidence", ""),
            ])

    overall_elapsed = time.time() - overall_start
    log(f"\n{'='*60}")
    log(f"Kept images   : {total_kept_images}")
    log(f"Dropped images: {total_dropped_images}")
    log(f"Kept detections (boxes): {total_kept_detections}")
    if run_ocr:
        gps_rate = (gps_ocr_successes / gps_ocr_attempts * 100) if gps_ocr_attempts else 0.0
        log(f"GPS OCR success rate: {gps_ocr_successes}/{gps_ocr_attempts} ({gps_rate:.1f}%) "
            f"-- check manifest.json gps fields before trusting downstream")
    log(f"Total time: {overall_elapsed/60:.1f} min")
    log(f"Images (plain)     : {images_out}")
    log(f"Images (annotated) : {annotated_out}")
    log(f"Labels (YOLO .txt) : {labels_out}")
    log(f"Manifest (json)    : {manifest_path}")
    log(f"Metadata (csv, GPS only) : {metadata_csv_path}")
    log(f"NOTE: real_world_length_m is null for every detection -- bbox_diagonal_px")
    log(f"      is a PIXEL measurement only. See script docstring before using it")
    log(f"      as a physical crack length.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="A single video file, a folder of videos, OR a folder of pre-extracted images")
    ap.add_argument("--no-gps-ocr", action="store_true", help="Skip GPS OCR entirely (faster; use if your footage has no on-screen GPS overlay)")
    args = ap.parse_args()

    input_path = Path(args.input).expanduser()

    if not MODEL_PATH.exists():
        raise SystemExit(
            f"MODEL_PATH does not exist: {MODEL_PATH}\n"
            f"Edit the MODEL_PATH constant near the top of this script to point "
            f"at your actual trained weights."
        )

    input_type, video_files_or_image_dir = detect_input_type(input_path)
    log(f"Input type detected: {input_type}")

    # label used to build the auto-numbered output folder name, e.g.
    # "GRMN0015.MP4" for a single video file, or the folder's own name for
    # a folder of videos/images
    label = input_path.name if input_path.is_file() else (input_path.name or "input")
    out_dir = next_numbered_output_dir(OUTPUT_BASE_DIR, label)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output folder: {out_dir}")

    batch_tag = re.sub(r"[^A-Za-z0-9]", "", input_path.stem)[:20] or "batch"

    if input_type == "video":
        frames_dir = out_dir / "frames"
        extract_frames(video_files_or_image_dir, frames_dir, batch_tag, fps_extract=FPS_EXTRACT)
    else:
        log("Input is already images -- skipping frame extraction.")
        frames_dir = video_files_or_image_dir

    track_and_filter(
        frames_dir=frames_dir,
        model_path=MODEL_PATH,
        output_dir=out_dir,
        batch_tag=batch_tag,
        min_track_length=MIN_TRACK_LENGTH,
        conf_floor=CONF_FLOOR,
        max_gap=MAX_GAP,
        samples_per_run=SAMPLES_PER_RUN,
        run_ocr=not args.no_gps_ocr,
    )


if __name__ == "__main__":
    main()
