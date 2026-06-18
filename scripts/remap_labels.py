import os
import shutil
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
SRC_LABELS = Path("/homes/kalluri1/Pavement_Distress_Detection/Dataset/full/labels")
SRC_IMAGES = Path("/homes/kalluri1/Pavement_Distress_Detection/Dataset/full/images")
DST_ROOT   = Path("/homes/kalluri1/Pavement_Distress_Detection/Dataset/full_remapped")
DST_LABELS = DST_ROOT / "labels"
DST_IMAGES = DST_ROOT / "images"

# ── Class mapping (old ID → new ID, -1 = drop) ─────────────────────────────
REMAP = {
    0:  0,   # Longitudinal Crack      → Longitudinal Crack
    1:  1,   # Transverse Crack        → Transverse Crack
    2:  2,   # Alligator Crack         → Alligator Crack
    3:  3,   # Pothole                 → Pothole
    4: -1,   # Manhole Cover           → DROP
    5:  4,   # Longitudinal Patch      → Patch
    6:  4,   # Transverse Patch        → Patch
    7:  5,   # Block Crack             → Block Crack
    8:  0,   # Sealed Longitudinal     → Longitudinal Crack
    9:  1,   # Sealed Transverse       → Transverse Crack
    10: 0,   # Line Crack              → Longitudinal Crack
    11: 1,   # Oblique Crack           → Transverse Crack
    12: 4,   # Repair                  → Patch
    13: 0,   # Lateral Crack           → Longitudinal Crack
    14: -1,  # Other Corruption        → DROP
}

NEW_CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Pothole",
    "Patch",
    "Block Crack",
]

# ── Setup ───────────────────────────────────────────────────────────────────
DST_LABELS.mkdir(parents=True, exist_ok=True)
DST_IMAGES.mkdir(parents=True, exist_ok=True)

# ── Stats ───────────────────────────────────────────────────────────────────
stats = {i: 0 for i in range(6)}
dropped = 0
skipped_files = 0
processed_files = 0

label_files = list(SRC_LABELS.glob("*.txt"))
total = len(label_files)
print(f"Found {total} label files. Starting remap...")

for i, lf in enumerate(label_files):
    if i % 5000 == 0:
        print(f"  {i}/{total} processed...")

    new_lines = []
    with open(lf, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        old_id = int(parts[0])
        new_id = REMAP.get(old_id, -1)
        if new_id == -1:
            dropped += 1
            continue
        stats[new_id] += 1
        new_lines.append(f"{new_id} {' '.join(parts[1:])}")

    # Write label file even if empty (image has no valid annotations)
    dst_lf = DST_LABELS / lf.name
    with open(dst_lf, "w") as f:
        f.write("\n".join(new_lines))

    # Symlink the image instead of copying (saves disk space)
    stem = lf.stem
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        src_img = SRC_IMAGES / (stem + ext)
        if src_img.exists():
            dst_img = DST_IMAGES / (stem + ext)
            if not dst_img.exists():
                os.symlink(src_img, dst_img)
            break

    processed_files += 1

# ── Write new classes.txt ───────────────────────────────────────────────────
with open(DST_ROOT / "classes.txt", "w") as f:
    for cls in NEW_CLASSES:
        f.write(cls + "\n")

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n=== REMAP COMPLETE ===")
print(f"Label files processed : {processed_files}")
print(f"Annotations dropped   : {dropped} (Manhole Cover + Other Corruption)")
print(f"\nNew class distribution:")
for idx, name in enumerate(NEW_CLASSES):
    print(f"  {idx}: {name:<25} {stats[idx]:>8} annotations")
print(f"\nOutput: {DST_ROOT}")
