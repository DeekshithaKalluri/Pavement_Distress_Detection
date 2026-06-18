import cv2
from pathlib import Path

VIDEO_DIR = Path.home() / "Pavement_Distress_Detection" / "Dataset" / "vid1"
OUTPUT_DIR = Path.home() / "Pavement_Distress_Detection" / "Dataset" / "frames" / "vid1"
BATCH_TAG = VIDEO_DIR.name  # e.g. "vid1" -- prefixed onto every frame filename so that
                            # Garmin's recycled filenames (e.g. GRMN0015.MP4 appearing in
                            # both vid1/ and a future vid2/) never produce colliding output
                            # filenames. Change VIDEO_DIR/OUTPUT_DIR to "vid2", "vid3", etc.
                            # for future batches -- BATCH_TAG updates automatically.
FPS_EXTRACT = 4  # frames per second to extract
BAR_WIDTH = 20
BAR_CHECKPOINTS = 10


def make_bar(fraction):
    filled = int(round(fraction * BAR_WIDTH))
    filled = max(0, min(BAR_WIDTH, filled))
    return "[" + "#" * filled + "-" * (BAR_WIDTH - filled) + "]"


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

video_files = sorted(VIDEO_DIR.glob("*.MP4")) + sorted(VIDEO_DIR.glob("*.mp4"))
print(f"Found {len(video_files)} videos in {VIDEO_DIR}", flush=True)

total_frames_saved = 0

for video_num, video_path in enumerate(video_files, start=1):
    print(f"[{video_num}/{len(video_files)}] Starting: {video_path.name}", flush=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  WARNING: could not open {video_path.name}, skipping", flush=True)
        continue

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0:
        native_fps = 30.0

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, round(native_fps / FPS_EXTRACT))

    video_stem = video_path.stem
    frame_idx = 0
    saved_idx = 0
    next_checkpoint = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            out_name = f"{BATCH_TAG}_{video_stem}_f{saved_idx:05d}.jpg"
            out_path = OUTPUT_DIR / out_name
            cv2.imwrite(str(out_path), frame)
            saved_idx += 1

        frame_idx += 1

        if total_video_frames > 0:
            progress_fraction = frame_idx / total_video_frames
            checkpoint_fraction = next_checkpoint / BAR_CHECKPOINTS
            if progress_fraction >= checkpoint_fraction and next_checkpoint <= BAR_CHECKPOINTS:
                pct = int(round(progress_fraction * 100))
                bar = make_bar(progress_fraction)
                print(
                    f"  [{video_num}/{len(video_files)}] {video_path.name}: {bar} {pct}% "
                    f"({frame_idx}/{total_video_frames} frames read, {saved_idx} saved)",
                    flush=True,
                )
                next_checkpoint += 1

    cap.release()
    total_frames_saved += saved_idx
    print(
        f"[{video_num}/{len(video_files)}] Done: {video_path.name} -- native_fps={native_fps:.1f}, "
        f"interval={frame_interval}, saved {saved_idx} frames (running total: {total_frames_saved})",
        flush=True,
    )

print(f"\nTotal frames extracted: {total_frames_saved}", flush=True)
print(f"Saved to: {OUTPUT_DIR}", flush=True)