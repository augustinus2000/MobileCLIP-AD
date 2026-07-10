"""
MPDD test dataset loader for dense anomaly localization.

This module builds image/mask/label samples from the MVTec AD test split.
Good samples receive zero masks, while anomalous samples load the
corresponding ground-truth defect masks.
"""

from pathlib import Path
from typing import List, Dict, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


class MPDDDenseDataset(Dataset):
    """
    MVTec AD test loader.

    Expected structure:
        root/class_name/test/good/*.png
        root/class_name/test/defect_type/*.png
        root/class_name/ground_truth/defect_type/*_mask.png

    For good images, mask is all zeros.
    For anomaly images:
        test/broken_large/000.png -> ground_truth/broken_large/000_mask.png
    """

    def __init__(
        self,
        root: str,
        image_size: int = 256,
        classes: Optional[List[str]] = None,
    ):
        self.root = Path(root)
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

        if classes is None:
            classes = sorted([
                p.name for p in self.root.iterdir()
                if p.is_dir() and (p / "test").exists()
            ])

        self.classes = classes
        self.samples: List[Dict] = []
        self._build_index()

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {self.root}")

    def _is_image(self, p: Path) -> bool:
        return p.is_file() and p.suffix in IMG_EXTS

    def _build_index(self):
        for cls in self.classes:
            cls_dir = self.root / cls
            test_root = cls_dir / "test"
            gt_root = cls_dir / "ground_truth"

            if not test_root.exists():
                continue

            for defect_dir in sorted([p for p in test_root.iterdir() if p.is_dir()]):
                defect_type = defect_dir.name
                is_good = defect_type == "good"

                for img_path in sorted(defect_dir.rglob("*")):
                    if not self._is_image(img_path):
                        continue

                    if is_good:
                        mask_path = None
                        label = 0
                    else:
                        mask_path = gt_root / defect_type / f"{img_path.stem}_mask.png"
                        if not mask_path.exists():
                            candidates = list((gt_root / defect_type).glob(f"{img_path.stem}*"))
                            mask_path = candidates[0] if candidates else None
                        label = 1

                    self.samples.append({
                        "image_path": img_path,
                        "mask_path": mask_path,
                        "class_name": cls,
                        "label": label,
                        "defect_type": defect_type,
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
    ds = MPDDDenseDataset("../datasets/mpdd_data/MPDD", image_size=256)
    print("num samples:", len(ds))
    print("classes:", ds.classes)
    x = ds[0]
    print(x["image"].shape, x["mask"].shape, x["label"], x["class_name"], x["defect_type"], x["image_path"])
