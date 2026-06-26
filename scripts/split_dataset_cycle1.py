import os
import random
import shutil
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
SRC     = Path("/homes/kalluri1/Pavement_Distress_Detection/Dataset/full_cycle1")
DST     = Path("/homes/kalluri1/Pavement_Distress_Detection/Dataset/split_cycle1")
TRAIN   = 0.80
VAL     = 0.10
TEST    = 0.10
SEED    = 42

# ── Setup ───────────────────────────────────────────────────────────────────
random.seed(SEED)

for split in ["train", "val", "test"]:
    (DST / split / "images").mkdir(parents=True, exist_ok=True)
    (DST / split / "labels").mkdir(parents=True, exist_ok=True)

# ── Gather all label files that have a matching image ───────────────────────
label_files = sorted(SRC.glob("labels/*.txt"))
valid_pairs = []

for lf in label_files:
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        img = SRC / "images" / (lf.stem + ext)
        if img.exists():
            valid_pairs.append((lf, img))
            break

print(f"Valid image-label pairs: {len(valid_pairs)}")

# ── Shuffle and split ────────────────────────────────────────────────────────
random.shuffle(valid_pairs)
n = len(valid_pairs)
n_train = int(n * TRAIN)
n_val   = int(n * VAL)

splits = {
    "train": valid_pairs[:n_train],
    "val":   valid_pairs[n_train:n_train + n_val],
    "test":  valid_pairs[n_train + n_val:],
}

# ── Symlink files into split folders ────────────────────────────────────────
for split_name, pairs in splits.items():
    print(f"Creating {split_name}: {len(pairs)} images...")
    for lf, img in pairs:
        dst_img = DST / split_name / "images" / img.name
        dst_lbl = DST / split_name / "labels" / lf.name
        if not dst_img.exists():
            os.symlink(img, dst_img)
        if not dst_lbl.exists():
            os.symlink(lf, dst_lbl)

# ── Write data.yaml ──────────────────────────────────────────────────────────
yaml_content = f"""path: {DST}
train: train/images
val: val/images
test: test/images

nc: 6
names:
  0: Longitudinal Crack
  1: Transverse Crack
  2: Alligator Crack
  3: Pothole
  4: Patch
  5: Block Crack
"""

with open(DST / "data.yaml", "w") as f:
    f.write(yaml_content)

# ── Copy classes.txt ─────────────────────────────────────────────────────────
shutil.copy(SRC / "classes.txt", DST / "classes.txt")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n=== SPLIT COMPLETE ===")
for split_name, pairs in splits.items():
    print(f"  {split_name:<6}: {len(pairs):>6} images")
print(f"\ndata.yaml written to: {DST}/data.yaml")
