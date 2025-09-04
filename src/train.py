# src/train.py
import argparse, json, time, os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from monai.losses import DiceCELoss

# Prefer new AMP API; fall back if PyTorch is older
try:
    from torch.amp import autocast, GradScaler
except Exception:
    from torch.cuda.amp import autocast, GradScaler  # fallback

# W&B is optional; guard the import
try:
    import wandb
except Exception:
    wandb = None

# local modules
from dataset import SegDataset
from models.unet import build_unet


# ------------------------- helpers ------------------------- #
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def dice_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int):
    """Mean Dice across classes 1..C-1 (background=0 excluded)."""
    dices = []
    for c in range(1, num_classes):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().item()
        denom = p.sum().item() + t.sum().item()
        if denom > 0:
            dices.append(2.0 * inter / denom)
    return float(np.mean(dices)) if dices else 0.0


def save_val_overlays(batch_imgs, batch_pred, run_dir: Path, tag: str, max_save: int = 4):
    """
    Save quick visualization PNGs and return the saved paths.
    batch_imgs: (N,1,H,W) float32 in [0,1]
    batch_pred: (N,H,W) int64
    """
    import cv2
    run_dir.mkdir(parents=True, exist_ok=True)
    n = min(batch_imgs.size(0), max_save)
    imgs = batch_imgs[:n].detach().cpu().numpy()
    preds = batch_pred[:n].detach().cpu().numpy()
    saved_paths = []
    for i in range(n):
        g = (imgs[i, 0] * 255).astype(np.uint8)
        g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        p = preds[i].astype(np.uint8)
        scale = max(1, int(p.max()))  # avoid div-by-zero
        color = cv2.applyColorMap((p * int(255 / scale)).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(g, 0.6, color, 0.4, 0)
        out_path = run_dir / f"{tag}_sample{i}.png"
        cv2.imwrite(str(out_path), overlay)
        saved_paths.append(out_path)
    return saved_paths


# ---------------------------- CLI -------------------------- #
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/processed/images", help="DICOM folder")
    ap.add_argument("--masks",  default="data/processed/masks_indexed", help="Indexed masks folder (0..7)")
    ap.add_argument("--splits", default="data/splits.json", help="JSON with train/val/test lists")
    ap.add_argument("--num_classes", type=int, default=8, help="background + 7 classes")
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision")
    ap.add_argument("--out", default="outputs/unet_baseline")

    # Weights & Biases (optional)
    ap.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    ap.add_argument("--wandb_project", default="ct-seg", help="W&B project name")
    ap.add_argument("--wandb_entity", default=None, help="W&B entity/team (optional)")
    ap.add_argument("--run_name", default=None, help="Optional run name (defaults to output dir name)")
    return ap.parse_args()


# ---------------------------- main ------------------------- #
def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ckpts").mkdir(parents=True, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(log_dir=str(run_dir))

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Optional: Weights & Biases (mirrors TensorBoard, plus images/artifacts)
    wb_run = None
    if args.wandb:
        if wandb is None:
            print("[WARN] --wandb was set but the 'wandb' package is not installed. Skipping W&B.")
        else:
            wb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.run_name or run_dir.name,
                config=vars(args),
                dir=str(run_dir),
                sync_tensorboard=True,  # mirror SummaryWriter scalars
            )

    # ---------- data ----------
    splits = json.load(open(args.splits))
    train_names, val_names = splits["train"], splits["val"]

    ds_tr = SegDataset(args.images, args.masks, train_names, img_size=args.img_size, augment=True)
    ds_va = SegDataset(args.images, args.masks, val_names,   img_size=args.img_size, augment=False)

    dl_tr = DataLoader(
        ds_tr, batch_size=args.bs, shuffle=True,
        num_workers=args.workers, pin_memory=(device_type == "cuda"), drop_last=True
    )
    dl_va = DataLoader(
        ds_va, batch_size=args.bs, shuffle=False,
        num_workers=args.workers, pin_memory=(device_type == "cuda")
    )

    # ---------- model/loss/opt ----------
    model = build_unet(in_channels=1, out_channels=args.num_classes).to(device)
    opt = AdamW(model.parameters(), lr=args.lr)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)

    # GradScaler only needed when CUDA + AMP (fp16); unnecessary for bf16/CPU
    scaler = GradScaler(enabled=(args.amp and device_type == "cuda"))

    print(f"Train={len(ds_tr)}  Val={len(ds_va)}  Device={device}")
    best_mdice = -1.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        t0 = time.time()
        tr_loss = 0.0

        for img, mask in dl_tr:
            img  = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True).long()
            if img.ndim == 3:             # safety: [B,H,W] -> [B,1,H,W]
                img = img.unsqueeze(1)
            mask1 = mask.unsqueeze(1)  
            opt.zero_grad(set_to_none=True)

            with autocast(device_type=device_type, enabled=args.amp):
                logits = model(img)             # (N,C,H,W)
                loss = loss_fn(logits, mask1)    # Dice + CE

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            bs = img.size(0)
            tr_loss += loss.item() * bs
            writer.add_scalar("loss/train_step", loss.item(), global_step)
            global_step += 1

        tr_loss /= max(1, len(ds_tr))
        writer.add_scalar("loss/train_epoch", tr_loss, epoch)

        # ---- validate ----
        model.eval()
        va_loss = 0.0
        mdices = []
        first_batch_imgs = None
        first_batch_pred = None

        with torch.no_grad():
            for i, (img, mask) in enumerate(dl_va):
                img  = img.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True).long()
                if img.ndim == 3:
                    img = img.unsqueeze(1)
                mask1 = mask.unsqueeze(1)

                with autocast(device_type=device_type, enabled=args.amp):
                    logits = model(img)
                    loss = loss_fn(logits, mask1)
                va_loss += loss.item() * img.size(0)

                pred = logits.softmax(1).argmax(1)  # (N,H,W)
                md = dice_per_class(pred, mask, args.num_classes)
                mdices.append(md)

                if first_batch_imgs is None:
                    first_batch_imgs = img.clone()
                    first_batch_pred = pred.clone()

        va_loss /= max(1, len(ds_va))
        mdice = float(np.mean(mdices)) if mdices else 0.0

        writer.add_scalar("loss/val", va_loss, epoch)
        writer.add_scalar("dice/mean_no_bg", mdice, epoch)

        # Save a few overlay images and log to W&B if enabled
        try:
            paths = save_val_overlays(first_batch_imgs, first_batch_pred, run_dir / "val_vis", f"epoch{epoch:03d}")
            if wb_run is not None and paths:
                wandb.log({
                    "val/overlays": [wandb.Image(str(p), caption=p.name) for p in paths],
                    "epoch": epoch
                })
        except Exception as e:
            print("WARN: could not save/log overlays:", e)

        dt = time.time() - t0
        print(f"[{epoch:03d}/{args.epochs}] "
              f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  mdice={mdice:.4f}  ({dt:.1f}s)")

        # Save best checkpoint + W&B artifact
        if mdice > best_mdice:
            best_mdice = mdice
            ckpt_path = run_dir / "ckpts" / "best.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch, "mdice": mdice}, ckpt_path)
            with open(run_dir / "best.txt", "w") as f:
                f.write(json.dumps({"epoch": epoch, "mdice": best_mdice}, indent=2))
            if wb_run is not None:
                try:
                    art = wandb.Artifact(
                        name=f"{run_dir.name}-best",
                        type="model",
                        metadata={"epoch": epoch, "mdice": mdice}
                    )
                    art.add_file(str(ckpt_path))
                    wandb.log_artifact(art)
                except Exception as e:
                    print("WARN: could not log W&B artifact:", e)

    print(f"Best mean Dice (no bg): {best_mdice:.4f}")
    writer.close()
    if wb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
