
import os, glob, random
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# Try imagecorruptions for rich corruptions; fallback if unavailable
_HAS_IMGCORR = False
try:
    from imagecorruptions import corrupt, get_corruption_names
    _HAS_IMGCORR = True
except Exception:
    _HAS_IMGCORR = False

def _pil_add_gaussian_noise(img: Image.Image, std=0.08):
    arr = np.asarray(img).astype(np.float32) / 255.0
    noise = np.random.normal(0.0, std, arr.shape).astype(np.float32)
    out   = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((out * 255).astype(np.uint8))

def _pil_salt_pepper(img: Image.Image, amt=0.02):
    arr = np.asarray(img).copy()
    num = int(amt * arr.shape[0] * arr.shape[1])
    coords = (np.random.randint(0, arr.shape[0], num),
              np.random.randint(0, arr.shape[1], num))
    arr[coords] = 255  # salt
    coords = (np.random.randint(0, arr.shape[0], num),
              np.random.randint(0, arr.shape[1], num))
    arr[coords] = 0    # pepper
    return Image.fromarray(arr)

def _pil_jpeg(img: Image.Image, quality=30):
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def _pil_gaussian_blur(img: Image.Image, radius=2):
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

_FALLBACKS = [
    ("gaussian_noise", lambda im: _pil_add_gaussian_noise(im, std=0.08)),
    ("salt_pepper",    lambda im: _pil_salt_pepper(im, amt=0.02)),
    ("jpeg",           lambda im: _pil_jpeg(im, quality=30)),
    ("blur",           lambda im: _pil_gaussian_blur(im, radius=2)),
    ("brightness",     lambda im: ImageEnhance.Brightness(im).enhance(0.6)),
]

def apply_corruption(img: Image.Image, corruption_name: Optional[str]=None, severity: int=3) -> Image.Image:
    img = img.convert("RGB")
    if _HAS_IMGCORR:
        if corruption_name is None:
            corruption_name = random.choice(get_corruption_names())
        arr = np.array(img)
        arr_cor = corrupt(arr, corruption_name=corruption_name, severity=severity)
        return Image.fromarray(arr_cor)
    else:
        if corruption_name is None:
            _, fn = random.choice(_FALLBACKS)
        else:
            match = [fn for (n, fn) in _FALLBACKS if n == corruption_name]
            fn = match[0] if match else random.choice(_FALLBACKS)[1]
        return fn(img)

class PairedCorruptionListDataset(Dataset):
    """Yield (X_corrupt, X_clean) pairs from explicit image paths."""
    def __init__(self, img_paths: List[str], image_size=128, severity=3, corruption_name=None):
        self.paths = img_paths
        self.size  = image_size
        self.severity = severity
        self.corruption_name = corruption_name
        self.to_tensor = T.ToTensor()

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        clean = self.to_tensor(img)
        corrupt_img = apply_corruption(img, self.corruption_name, severity=self.severity)
        corrupt_img = corrupt_img.resize((self.size, self.size), Image.BILINEAR)
        corrupt = self.to_tensor(corrupt_img)
        return corrupt, clean
