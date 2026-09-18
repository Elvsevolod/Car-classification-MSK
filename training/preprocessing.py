"""Shared deterministic geometry for stage-3 training and evaluation."""
import numpy as np
from PIL import Image

from backend.core import crop_image


IMAGE_SIZE = 208
IMAGENET_MEAN = np.array([.485, .456, .406], dtype=np.float32)
IMAGENET_STD = np.array([.229, .224, .225], dtype=np.float32)
LETTERBOX_FILL = tuple(int(round(value * 255)) for value in IMAGENET_MEAN)


def resize_crop(crop, mode, size=IMAGE_SIZE):
    """Resize a PIL crop either by distortion or aspect-ratio preserving padding."""
    if mode == "square":
        return crop.resize((size, size), Image.Resampling.BILINEAR)
    if mode != "letterbox":
        raise ValueError("resize mode must be 'square' or 'letterbox'")
    scale = min(size / crop.width, size / crop.height)
    width = max(1, round(crop.width * scale))
    height = max(1, round(crop.height * scale))
    resized = crop.resize((width, height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), LETTERBOX_FILL)
    canvas.paste(resized, ((size - width) // 2, (size - height) // 2))
    return canvas


def preprocess_mode(image, box, mode):
    crop = crop_image(image, box)
    pixels = np.asarray(resize_crop(crop, mode), dtype=np.float32) / np.float32(255)
    pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(pixels.transpose(2, 0, 1))


class ResizeCrop:
    def __init__(self, mode="square", size=IMAGE_SIZE):
        if mode not in {"square", "letterbox"}:
            raise ValueError("resize mode must be 'square' or 'letterbox'")
        self.mode = mode
        self.size = size

    def __call__(self, image):
        return resize_crop(image, self.mode, self.size)
