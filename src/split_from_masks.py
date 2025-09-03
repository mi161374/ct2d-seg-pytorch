# src/split_from_masks.py
import argparse, json
from pathlib import Path
import random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True, help="data/processed/masks_indexed")
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--out", default="data/splits.json")
    args = ap.parse_args()

    names = [p.name for p in sorted(Path(args.masks).glob("*.png"))]
    random.seed(42); random.shuffle(names)
    n = len(names)
    n_tr = int(n * args.train)
    n_va = int(n * args.val)
    split = {
        "train": names[:n_tr],
        "val":   names[n_tr:n_tr+n_va],
        "test":  names[n_tr+n_va:]
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f: json.dump(split, f, indent=2)
    print(f"Wrote {args.out} | train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")

if __name__ == "__main__":
    main()
