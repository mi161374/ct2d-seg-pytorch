# src/merge_masks.py
import argparse, os, cv2, numpy as np
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="DICOM dir: data/raw/dicoms_n")
    ap.add_argument("--labels_root", required=True, help="Root of class folders: data/raw/masks")
    ap.add_argument("--out", required=True, help="Output dir for indexed masks, e.g., data/processed/masks_indexed")
    ap.add_argument("--classes", nargs="*", default=None, help="Optional explicit class folder order")
    ap.add_argument("--mask_prefix", default="", help="If masks are like <prefix><stem><suffix>.png")
    ap.add_argument("--mask_suffix", default="", help="If masks are like <prefix><stem><suffix>.png")
    ap.add_argument("--skip_empty", action="store_true", help="Skip saving if no class pixels")
    return ap.parse_args()

def main():
    args = parse_args()
    images_dir = Path(args.images)
    labels_root = Path(args.labels_root)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # class folders
    if args.classes:
        class_dirs = [labels_root / c for c in args.classes]
    else:
        class_dirs = sorted([p for p in labels_root.iterdir() if p.is_dir()])
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
            cand = cls_dir / f"{args.mask_prefix}{stem}{args.mask_suffix}.png"
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
            # no masks at all → either skip or write all-background
            if args.skip_empty:
                skipped += 1
                continue
            else:
                # make a blank same size as the first available class example, or 512x512 fallback
                # (practically, --skip_empty is recommended)
                indexed = np.zeros((512, 512), np.uint8)

        if args.skip_empty and not any_pixels:
            skipped += 1
            continue

        cv2.imwrite(str(out_dir / f"{stem}.png"), indexed)
        saved += 1

    print(f"Saved {saved} masks to {out_dir}. Skipped {skipped}.")
    print("Missing per class (file not found for that stem):")
    for k, v in missing_counts.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
