"""
Severstal Steel Defect dataset loader for dense anomaly localization.

Expected structure:
    root/train_images/*.jpg
    root/train.csv

train.csv columns:
    ImageId, ClassId, EncodedPixels

We evaluate on train_images because Kaggle test labels are not public.
Images without RLE annotations are treated as normal.
Multiple defect-class RLE masks are merged into one binary anomaly mask.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class SeverstalDenseDataset(Dataset):
    def __init__(self, root: str, image_size: int = 256):
        self.root = Path(root)
        self.image_dir = self.root / "train_images"
        self.csv_path = self.root / "train.csv"
        self.image_size = image_size

        self.image_tf = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

        self.mask_tf = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

        self.classes = ["severstal"]
        self.rle_map: Dict[str, List[str]] = {}
        self.class_map: Dict[str, List[str]] = {}

        df = pd.read_csv(self.csv_path)
        for image_id, g in df.groupby("ImageId"):
            self.rle_map[image_id] = [str(x) for x in g["EncodedPixels"].tolist()]
            self.class_map[image_id] = [str(x) for x in g["ClassId"].tolist()]

        self.samples = []
        for img_path in sorted(self.image_dir.glob("*.jpg")):
            image_id = img_path.name
            has_defect = image_id in self.rle_map
            defect_types = self.class_map.get(image_id, [])
            self.samples.append({
                "image_path": img_path,
                "image_id": image_id,
                "label": 1 if has_defect else 0,
                "class_name": "severstal",
                "defect_type": "+".join(defect_types) if defect_types else "good",
                "rles": self.rle_map.get(image_id, []),
            })

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {self.image_dir}")

    def _decode_rle(self, rle: str, height: int, width: int) -> np.ndarray:
        mask = np.zeros(height * width, dtype=np.uint8)
        if rle is None or rle == "" or rle == "nan":
            return mask.reshape((height, width), order="F")

        vals = np.asarray([int(x) for x in rle.split()], dtype=np.int64)
        starts = vals[0::2] - 1
        lengths = vals[1::2]
        ends = starts + lengths

        for s, e in zip(starts, ends):
            mask[s:e] = 1

        return mask.reshape((height, width), order="F")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        img_pil = Image.open(item["image_path"]).convert("RGB")
        width, height = img_pil.size

        image = self.image_tf(img_pil)

        if item["label"] == 0:
            mask_np = np.zeros((height, width), dtype=np.uint8)
        else:
            mask_np = np.zeros((height, width), dtype=np.uint8)
            for rle in item["rles"]:
                mask_np |= self._decode_rle(rle, height=height, width=width)

        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
        mask = self.mask_tf(mask_pil)
        mask = (mask > 0.0).float()

        return {
            "image": image,
            "mask": mask,
            "label": torch.tensor(float(item["label"]), dtype=torch.float32),
            "class_name": item["class_name"],
            "defect_type": item["defect_type"],
            "image_path": str(item["image_path"]),
            "mask_path": "",
        }


if __name__ == "__main__":
    ds = SeverstalDenseDataset("../datasets_extra/severstal", image_size=256)
    n_anom = sum(int(s["label"]) for s in ds.samples)
    print("num samples:", len(ds))
    print("normal:", len(ds) - n_anom, "anomaly:", n_anom)
    x = ds[0]
    print(x["image"].shape, x["mask"].shape, x["label"], x["class_name"], x["defect_type"], x["image_path"])
