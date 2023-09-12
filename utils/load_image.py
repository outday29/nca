from PIL import Image
import torch
import numpy as np

def load_image(img_path, size=64, thumbnail_size=32):
    # From https://github.com/shyamsn97/controllable-ncas/blob/master/controllable_nca/utils.py
    # TODO: Need to support non-square grid
    img = Image.open(img_path)
    img.thumbnail((thumbnail_size, thumbnail_size), Image.Resampling.LANCZOS)
    img = np.float32(img) / 255.0
    # premultiply RGB by Alpha
    img[..., :3] *= img[..., 3:]
    # pad to self.h, self.h
    diff = size - thumbnail_size
    img = torch.tensor(img).permute(2, 0, 1)
    img = torch.nn.functional.pad(
        img, [diff // 2, diff // 2, diff // 2, diff // 2], mode="constant", value=0
    )
    return img
