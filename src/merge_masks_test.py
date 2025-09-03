# src/merge_masks.py
import argparse, cv2, numpy as np
from pathlib import Path

def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="DICOM dir: data/raw/images")
    ap.add_argument("--labels_root", required=True, help="Root of class folders: data/raw/labels")
    ap.add_argument("--out", required=True, help="Output dir for indexed masks, e.g., data/processed/masks_indexed")
    ap.add_argument("--classes", nargs="*", default=None, help="Optional explicit class folder order")
    ap.add_argument("--mask_prefix", default="", help="If masks are like <prefix><stem><suffix>.png")
    ap.add_argument("--mask_suffix", default="", help="If masks are like <prefix><stem><suffix>.png")
    ap.add_argument("--skip_empty", action="store_true", help="Skip saving if no class pixels")
    return ap.parse_args(argv)

def run_merge(images, labels_root, out, classes=None, mask_prefix="", mask_suffix="", skip_empty=False):
    # ⬇️ use the FUNCTION PARAMETERS, not argparse
    images_dir  = Path(images)
    labels_root = Path(labels_root)
    out_dir = Path(out); out_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = [labels_root / c for c in classes] if classes else \
                 sorted([p for p in labels_root.iterdir() if p.is_dir()])
    class_dirs = [p for p in class_dirs if p.exists()]
    if not class_dirs:
        raise SystemExit(f"No class folders under {labels_root}")

    dcm_stems = [p.stem for p in sorted(images_dir.glob("*.dcm"))]
    if not dcm_stems:
        raise SystemExit(f"No DICOMs found in {images_dir}")

    print("Class order (last wins):", [p.name for p in class_dirs])

    saved, skipped = 0, 0
    missing_counts = {p.name: 0 for p in class_dirs}

    for stem in dcm_stems:
        indexed = None
        any_pixels = False
        for cls_id, cls_dir in enumerate(class_dirs, start=1):
            cand = cls_dir / f"{mask_prefix}{stem}{mask_suffix}.png"
            if not cand.exists():
                missing_counts[cls_dir.name] += 1
                continue
            m = cv2.imread(str(cand), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            if indexed is None:
                indexed = np.zeros_like(m, dtype=np.uint8)
            pos = m > 0
            if pos.any():
                any_pixels = True
                indexed[pos] = cls_id  # overwrite for priority

        if indexed is None:
            if skip_empty:
                skipped += 1
                continue
            indexed = np.zeros((512, 512), np.uint8)  # fallback

        if skip_empty and not any_pixels:
            skipped += 1
            continue

        cv2.imwrite(str(out_dir / f"{stem}.png"), indexed)
        saved += 1

    print(f"Saved {saved} masks to {out_dir}. Skipped {skipped}.")
    print("Missing per class (file not found for that stem):")
    for k, v in missing_counts.items():
        print(f"  {k}: {v}")

def main():
    args = parse_args()
    run_merge(**vars(args))
