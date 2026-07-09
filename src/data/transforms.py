"""Albumentations transforms for BCC tiles: HED-space stain jitter and train/eval pipelines."""

import albumentations as A
import numpy as np
from albumentations import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2
from skimage.color import hed2rgb, rgb2hed

_INPUT_SIZE = 380
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class HEDJitter(ImageOnlyTransform):
    """Per-channel multiplicative+additive jitter in HED stain space for stained tiles."""

    def __init__(self, sigma: float = 0.05, bias: float = 0.05, p: float = 0.8):
        super().__init__(p=p)
        self.sigma = sigma
        self.bias = bias

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        if img.dtype != np.uint8:
            raise ValueError(f"HEDJitter expects uint8 input, got dtype {img.dtype}")
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"HEDJitter expects shape (H, W, 3), got {img.shape}")
        img_float = img.astype(np.float32) / 255.0
        hed = rgb2hed(img_float)
        sigmas = np.random.uniform(-self.sigma, self.sigma, size=3).astype(np.float32)
        biases = np.random.uniform(-self.bias, self.bias, size=3).astype(np.float32)
        hed_aug = hed * (1.0 + sigmas) + biases
        rgb_aug = np.clip(hed2rgb(hed_aug), 0.0, 1.0)
        return (rgb_aug * 255.0).astype(np.uint8)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("sigma", "bias")


def get_train_transforms(correction: bool) -> A.Compose:
    """Train pipeline; correction=True uses a random resized crop, False a fixed resize."""
    first = (
        A.RandomResizedCrop(
            size=(_INPUT_SIZE, _INPUT_SIZE), scale=(0.7, 1.0), ratio=(0.95, 1.05), p=1.0
        )
        if correction
        else A.Resize(_INPUT_SIZE, _INPUT_SIZE)
    )
    return A.Compose(
        [
            first,
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            HEDJitter(sigma=0.05, bias=0.05, p=0.8),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_eval_transforms() -> A.Compose:
    """Eval/inference pipeline: fixed resize, normalize, to tensor."""
    return A.Compose(
        [
            A.Resize(_INPUT_SIZE, _INPUT_SIZE),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    )
