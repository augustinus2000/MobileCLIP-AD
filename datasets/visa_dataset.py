"""
VisA training dataset loader for DenseAlign.

This module loads normal and anomalous VisA samples with pixel-level masks.
It is used to train the lightweight DenseAlign head while keeping
MobileCLIP2 frozen.
"""

import os
import csv
from pathlib import Path
from typing import List, Dict, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


class VisADenseDataset(Dataset):
    """
    VisA dataset loader for dense anomaly segmentation.

    Expected structure:
        root/class_name/Data/Images/Normal/*.JPG
        root/class_name/Data/Images/Anomaly/*.JPG
        root/class_name/Data/Masks/Anomaly/*.png

    For normal images, mask is all zeros.
    For anomaly images, mask is paired by stem:
        Images/Anomaly/001.JPG -> Masks/Anomaly/001.png
    """

    def __init__(
        self,
        root: str,
        image_size: int = 256,
        classes: Optional[List[str]] = None,
        use_normal: bool = True,
        use_anomaly: bool = True,
        split: str = "all",
        split_csv: Optional[str] = None,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.use_normal = use_normal
        self.use_anomaly = use_anomaly
        self.split = split
        self.split_csv = Path(split_csv) if split_csv is not None else None

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

        if classes is None:
            classes = sorted([
                p.name for p in self.root.iterdir()
                if p.is_dir() and (p / "Data" / "Images").exists()
            ])

        self.classes = classes
        self.samples: List[Dict] = []
        self._build_index()

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {self.root}")

    def _is_image(self, p: Path) -> bool:
        return p.is_file() and p.suffix in IMG_EXTS

    def _build_index(self):
        # Official VisA split mode using split_csv/1cls.csv.
        # Expected columns: object,split,label,image,mask
        if self.split_csv is not None:
            if not self.split_csv.exists():
                raise FileNotFoundError(f"split_csv not found: {self.split_csv}")

            with open(self.split_csv, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            valid_classes = set(self.classes) if self.classes is not None else None
            discovered_classes = []

            for row in rows:
                cls = row["object"]
                sp = row["split"]
                label_name = row["label"]

                if valid_classes is not None and cls not in valid_classes:
                    continue

                if self.split not in ("all", "full") and sp != self.split:
                    continue

                is_anomaly = label_name.lower() == "anomaly"
                is_normal = label_name.lower() == "normal"

                if is_normal and not self.use_normal:
                    continue
                if is_anomaly and not self.use_anomaly:
                    continue

                img_path = self.root / row["image"]
                mask_value = row.get("mask", "")

                if is_normal:
                    mask_path = None
                    label = 0
                    defect_type = "Normal"
                else:
                    mask_path = self.root / mask_value if mask_value else None
                    label = 1
                    defect_type = "Anomaly"

                if not img_path.exists():
                    raise FileNotFoundError(f"VisA image not found: {img_path}")

                if mask_path is not None and not mask_path.exists():
                    raise FileNotFoundError(f"VisA mask not found: {mask_path}")

                self.samples.append({
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "class_name": cls,
                    "label": label,
                    "defect_type": defect_type,
                })

                discovered_classes.append(cls)

            self.classes = sorted(set(discovered_classes))
            return

        # Legacy full-folder scan mode.
        for cls in self.classes:
            cls_dir = self.root / cls
            img_root = cls_dir / "Data" / "Images"
            mask_root = cls_dir / "Data" / "Masks"

            if self.use_normal:
                normal_dir = img_root / "Normal"
                if normal_dir.exists():
                    for img_path in sorted(normal_dir.rglob("*")):
                        if not self._is_image(img_path):
                            continue
                        self.samples.append({
                            "image_path": img_path,
                            "mask_path": None,
                            "class_name": cls,
                            "label": 0,
                            "defect_type": "Normal",
                        })

            if self.use_anomaly:
                anomaly_dir = img_root / "Anomaly"
                anomaly_mask_dir = mask_root / "Anomaly"

                if anomaly_dir.exists():
                    for img_path in sorted(anomaly_dir.rglob("*")):
                        if not self._is_image(img_path):
                            continue

                        mask_path = anomaly_mask_dir / f"{img_path.stem}.png"
                        if not mask_path.exists():
                            candidates = list(anomaly_mask_dir.glob(f"{img_path.stem}.*"))
                            mask_path = candidates[0] if candidates else None

                        self.samples.append({
                            "image_path": img_path,
                            "mask_path": mask_path,
                            "class_name": cls,
                            "label": 1,
                            "defect_type": "Anomaly",
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        img = Image.open(item["image_path"]).convert("RGB")
        image = self.image_tf(img)

        if item["mask_path"] is None:
            mask = torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)
        else:
            m = Image.open(item["mask_path"]).convert("L")
            mask = self.mask_tf(m)
            mask = (mask > 0.0).float()

        label = torch.tensor(float(item["label"]), dtype=torch.float32)

        return {
            "image": image,
            "mask": mask,
            "label": label,
            "class_name": item["class_name"],
            "defect_type": item["defect_type"],
            "image_path": str(item["image_path"]),
            "mask_path": str(item["mask_path"]) if item["mask_path"] is not None else "",
        }


if __name__ == "__main__":
    ds = VisADenseDataset("../datasets/visa", image_size=256)
    print("num samples:", len(ds))
    print("classes:", ds.classes)
    x = ds[0]
    print(x["image"].shape, x["mask"].shape, x["label"], x["class_name"], x["image_path"])
