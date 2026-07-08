"""Heidelberg BCC tile dataset for one split, backed by the shipped manifest."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

_VALID_SPLITS = ("Train", "Validation", "Test")


class HeidelbergBccDataset(Dataset):
    """Heidelberg H&E tiles for one split, labelled BCC (1) vs other (0) from the manifest.

    data_root must be the directory containing the extracted `data/` folder, since manifest
    `file` paths already begin with `data/`.
    """

    def __init__(self, manifest_path, data_root, split, transform=None) -> None:
        if split not in _VALID_SPLITS:
            raise ValueError(f"split must be one of {_VALID_SPLITS}; got {split!r}")
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform

        df = pd.read_csv(self.manifest_path, dtype={"case": str})
        if str(df.columns[0]).startswith("Unnamed:"):
            df = df.drop(columns=df.columns[0])
        self._frame = df[df["set"] == split].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._frame)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self._frame.iloc[idx]
        with Image.open(self.data_root / row["file"]) as img:
            image_arr = np.array(img.convert("RGB"))
        if self.transform is not None:
            image_tensor = self.transform(image=image_arr)["image"]
            if not isinstance(image_tensor, torch.Tensor):
                raise TypeError(
                    f"transform returned {type(image_tensor).__name__}; expected torch.Tensor. "
                    "Ensure the transform pipeline ends with albumentations.pytorch.ToTensorV2()."
                )
        else:
            image_tensor = torch.from_numpy(image_arr).permute(2, 0, 1).float() / 255.0
        label = torch.tensor([int(row["label"])], dtype=torch.float32)
        return image_tensor, label

    def get_imbalance_ratio(self) -> float:
        """Return n_negative / n_positive for this split."""
        is_pos = self._frame["label"] == 1
        n_pos = int(is_pos.sum())
        n_neg = int((~is_pos).sum())
        if n_pos == 0:
            raise ValueError(
                f"split {self.split!r} has zero positive (BCC) samples; "
                "cannot compute imbalance ratio."
            )
        return n_neg / n_pos

    def get_patient_ids(self) -> list[str]:
        """Unique `case` values for this split, sorted."""
        return sorted(self._frame["case"].astype(str).unique().tolist())
