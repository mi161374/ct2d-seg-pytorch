# src/dataset.py
import cv2, numpy as np, torch
from torch.utils.data import Dataset
from pathlib import Path

try:
    import pydicom
except Exception:
    pydicom = None

def _read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".dcm":
        assert pydicom is not None, "Install pydicom for DICOM support"
        d = pydicom.dcmread(str(path))
        img = d.pixel_array.astype(np.float32)
        slope = float(getattr(d, "RescaleSlope", 1.0))
        inter = float(getattr(d, "RescaleIntercept", 0.0))
        img = img * slope + inter      # HU-ish if CT
        img = np.clip(img, -1000, 1000)
        img = (img + 1000.0) / 2000.0  # → [0,1]
        return img
    # grayscale PNG/JPG
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise RuntimeError(f"Cannot read {path}")
    return (g.astype(np.float32) / 255.0)

class SegDataset(Dataset):
    def __init__(self, images_dir, masks_dir, names, img_size=512, augment=False):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.names = names
        self.img_size = img_size
        self.augment = augment

    def _find_image(self, name: str) -> Path:
        # try same basename with .png first, else .dcm
        png = self.images_dir / name
        dcm = self.images_dir / (Path(name).stem + ".dcm")
        if png.exists(): return png
        if dcm.exists(): return dcm
        raise FileNotFoundError(f"No image for {name}")

    def __len__(self): return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        ipath = self._find_image(name)
        mpath = self.masks_dir / name

        img = _read_image(ipath)  # HxW in [0,1], float32
        mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        if mask is None: raise RuntimeError(f"Cannot read mask {mpath}")

        # resize (keep NEAREST for masks)
        H, W = mask.shape
        if (H, W) != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # simple flips (optional)
        if self.augment:
            if np.random.rand() < 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
            if np.random.rand() < 0.5:
                img = np.flipud(img).copy()
                mask = np.flipud(mask).copy()

        # to tensors
        img_t = torch.from_numpy(img).unsqueeze(0)          # (1,H,W)
        mask_t = torch.from_numpy(mask.astype(np.int64))    # (H,W) long, indexed
        return img_t, mask_t
