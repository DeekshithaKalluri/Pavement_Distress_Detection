# Manual: How to Use the Pavement Distress Auto-Labeling Pipeline

## 1. Purpose

This pipeline automatically detects pavement distress (cracks, potholes,
patches, etc.) from road videos and generates YOLO-format labels, ready to
review or merge into a training set. It is a single combined script: give
it a video (or a folder of videos, or a folder of already-extracted images),
and it handles frame extraction, detection, temporal-consistency filtering,
GPS tagging, and annotated-image output in one run.

It replaces what used to be two separate steps (frame extraction, then
tracking/filtering) with one command.

---

## 2. Project Location

The pipeline lives at:

```bash
/homes/kalluri1/Pavement_Distress_Detection/
```

Main files:

```bash
auto_label_pipeline.py
scripts/run_autolabel_pipeline.sh
```

---

## 3. Input

You can point the pipeline at any of the following:

```bash
# a single video file
~/Pavement_Distress_Detection/Dataset/GRMN0015.MP4

# a folder containing multiple video files
~/Pavement_Distress_Detection/Dataset/vid2/

# a folder containing already-extracted images
~/Pavement_Distress_Detection/Dataset/frames/vid1/
```

Supported video formats: `.mp4`, `.MP4`, `.mov`, `.avi`, `.mkv`.

If the input is video, frames are extracted automatically at a fixed rate
(4 frames per second by default) before detection runs. If the input is
already a folder of images, frame extraction is skipped entirely and those
images are used directly.

A folder cannot contain both videos and images at the same time -- the
pipeline will stop and ask you to separate them, rather than guessing which
ones you meant to process.

---

## 4. Model File

The trained YOLO model path is fixed inside the script itself (not passed
on the command line), here:

```python
MODEL_PATH = Path.home() / "Pavement_Distress_Detection" / "runs" / "train" / "yolov11x" / "x.baseline" / "weights" / "best.pt"
```

If you retrain a new model and want the pipeline to use it going forward,
edit this one line in `auto_label_pipeline.py` and re-upload the file. This
is the only thing in the script you should normally need to change.

---

## 5. Output

Every run creates its own new, automatically numbered output folder, so
nothing ever gets overwritten between runs:

```bash
auto_labels/01_GRMN0015.MP4_output/
auto_labels/02_GRMN0016.MP4_output/
auto_labels/03_vid2_output/
```

The number always increases, even if you delete an earlier output folder.

Inside each output folder:

```bash
images/              -> kept frames, unmodified
images_annotated/    -> same frames, WITH bounding boxes + class name + confidence drawn on
labels/              -> YOLO-format .txt annotation files (class, box, confidence -- no GPS, see below)
manifest.json        -> one entry per kept image: GPS, and per-box pixel diagonal length
frames/              -> (video input only) the raw extracted frames before filtering
```

**Why GPS isn't in the `.txt` label files:** those stay in plain YOLO
format on purpose, so they can be used for retraining without any format
changes. GPS coordinates (read via OCR from the on-screen text burned into
the video by the dashcam) and the pixel-based box diagonal length are
written into `manifest.json` instead, alongside each kept image.

---

## 6. How the Auto-Labeling Logic Works

Unlike a simple "check every frame, keep the confident ones" approach, this
pipeline tracks each detected object across consecutive frames and only
keeps detections that persist -- the same way a real crack, viewed from a
moving vehicle, should keep being visible for more than one single frame.

For each video:

1. Frames are grouped back together by source video.
2. The YOLO model runs in **tracking mode**, which follows each detected
   object across frames rather than detecting each frame independently.
3. A detection is only kept if it forms a continuous run of at least
   `MIN_TRACK_LENGTH` frames in a row (default: 3), allowing for a small
   gap of up to `MAX_GAP` frames (default: 1) in case the object is briefly
   missed.
4. Each qualifying run is reduced down to `SAMPLES_PER_RUN` representative
   frames (default: 3 -- first, middle, last), rather than keeping every
   single frame of a long-lived detection. This avoids ending up with many
   near-duplicate images of the same crack.

These four numbers (`FPS_EXTRACT`, `MIN_TRACK_LENGTH`, `MAX_GAP`,
`CONF_FLOOR`, `SAMPLES_PER_RUN`) are set as constants near the top of
`auto_label_pipeline.py` if you ever need to tune them.

---

## 7. GPS Extraction (Read This Before Trusting It)

Some dashcams (including the Garmin units this project uses) burn GPS
coordinates directly into the video image as on-screen text, rather than
storing them as separate structured metadata. This pipeline reads that
on-screen text using OCR (optical character recognition) -- the same way a
person would read the corner of the screen -- and writes whatever it
successfully parses into `manifest.json`.

**This is not always reliable, and the pipeline does not pretend otherwise.**
Every run reports a GPS OCR success rate at the end, for example:

```bash
GPS OCR success rate: 0/11 (0.0%) -- check manifest.json gps fields before trusting downstream
```

A low or zero success rate can mean one of a few different things, and it's
worth checking which one before assuming GPS "isn't working":

- **The footage genuinely has no GPS lock yet.** Many dashcams don't start
  printing coordinates until some time after power-on, while they acquire
  a satellite signal. If your kept detections all happen to be near the
  start of the video, this is the most likely explanation.
- **The on-screen text isn't where the OCR step expects it.** The pipeline
  crops a strip near the bottom of the frame to read. If your camera's
  text overlay is positioned differently, sized differently, or styled
  differently than the footage this was built and tested against, the OCR
  step may be cropping the wrong region entirely.
- **OCR is reading the text but misreading some digits.** Check the
  `raw_ocr_text` field in `manifest.json` for a few entries -- if it shows
  garbled or partial text, OCR is seeing *something* but not reading it
  cleanly, which is a tuning problem (cropping/contrast), not a missing-GPS
  problem.

**Before trusting GPS for any video batch**, check a few `manifest.json`
entries directly:

```bash
cat auto_labels/<run_folder>/manifest.json
```

Look at the `gps` field for several kept images. If `lat`/`lon` are `null`
across the board, check `raw_ocr_text` for those same entries to tell which
of the three explanations above actually applies before assuming the
feature is broken or that the footage has no GPS at all.

If GPS isn't relevant to a given batch of footage at all, you can skip OCR
entirely (slightly faster) by adding `--no-gps-ocr` to the run:

```bash
sbatch scripts/run_autolabel_pipeline.sh ~/Pavement_Distress_Detection/Dataset/GRMN0015.MP4 --no-gps-ocr
```

---

## 8. Crack Length (Placeholder -- Not a Real Measurement Yet)

Each detection in `manifest.json` includes a `bbox_diagonal_px` value --
the diagonal length of the bounding box, **in pixels**. This is included
because it's a cheap, deterministic way to capture roughly how large a
detection's box is, and it requires no extra steps to compute.

**This is not a real-world measurement, and is not meant to be used as
one.** A box's pixel size depends on how far the actual crack is from the
camera and the camera's mounting height/angle/field of view -- a crack
right under the bumper and a crack the same physical size much farther
down the road will produce very different pixel diagonals. Converting a
pixel measurement into an actual physical length (e.g. centimeters or
meters) requires calibrating against the camera's known mounting geometry,
which has not been done for this project yet.

For this reason, every detection also includes a `real_world_length_m`
field, which is currently always `null`. It exists as a placeholder so
that, once camera calibration information becomes available, an actual
real-world length can be computed and filled in without needing to re-run
detection on the original footage -- the underlying box geometry needed to
do that calculation is already saved.

---

## 9. Running the Pipeline on Beocat GPU

**Do not run this script directly on the login/head node.** The head node
has no GPU, and the script will fail quickly with a CUDA error
(`Invalid CUDA 'device=0' requested`) rather than running slowly on CPU.
Always submit through the job scheduler.

The video/folder path is passed directly on the `sbatch` command line --
you do not need to open or edit any file to change which video gets
processed:

```bash
sbatch scripts/run_autolabel_pipeline.sh ~/Pavement_Distress_Detection/Dataset/GRMN0015.MP4
```

A different video next time -- still no file editing, just a different
argument:

```bash
sbatch scripts/run_autolabel_pipeline.sh ~/Pavement_Distress_Detection/Dataset/vid2
```

---

## 10. Monitoring a Run

Check whether your job is running:

```bash
squeue -u kalluri1
```

`ST` column shows `R` for running, `PD` for waiting in queue.

Watch live progress (find the job ID from `squeue` or from the `sbatch`
output when you submitted it):

```bash
tail -f ~/Pavement_Distress_Detection/logs/autolabel_pipeline_<JOBID>.log
```

`Ctrl+C` stops watching the log -- it does not stop the job itself.

If something goes wrong, check the separate error log:

```bash
cat ~/Pavement_Distress_Detection/logs/autolabel_pipeline_<JOBID>.err
```

---

## 11. Common Problems and Fixes

### Problem: `Invalid CUDA 'device=0' requested`, `torch.cuda.is_available(): False`

You ran the script directly instead of submitting it through `sbatch`. The
login node has no GPU. Submit through the scheduler as shown in Section 9.

### Problem: GPS fields are all `null` in `manifest.json`

See Section 7 in full -- check `raw_ocr_text` in the manifest before
concluding anything is broken. This can be a genuine "no GPS lock yet in
this footage" situation, not necessarily a bug.

### Problem: Very few images kept relative to how many frames were extracted

This is expected, and not necessarily a problem. Most extracted frames
will not contain a detection that persists long enough to survive temporal
filtering -- that's the filtering working as intended, not a failure. If
you suspect real distress is being lost (not just noise), it's worth
double-checking `MIN_TRACK_LENGTH` and `MAX_GAP` rather than assuming
something is wrong outright.

### Problem: No labeled output for a video at all

Check whether the video's frames are actually distinct YOLO-detectable
distress, or whether the road segment in that particular video may simply
not contain any of the six trained distress classes. Check the job's `.log`
file for that video's per-video summary line (`kept X, dropped Y`) to see
whether anything was kept at all versus whether tracking even produced raw
detections in the first place.

---

## 12. Summary

This pipeline takes raw dashcam video (or pre-extracted images) and
produces a clean, filtered, annotated set of pavement distress detections
in one command, with GPS context where available and a pixel-based box
size measurement reserved for future real-world calibration. It must be
run on a Beocat GPU node via `sbatch`, never directly on the login node,
and every run lands in its own clearly numbered output folder so nothing
ever gets overwritten.
