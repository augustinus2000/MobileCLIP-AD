"""
KSDD2 test dataset loader for dense anomaly localization.

Expected structure:
    root/test/*.png
    root/test/*_GT.png

For each image xxx.png, the corresponding mask is xxx_GT.png.
If the GT mask is empty, the image is treated as normal.
"""

from pathlib import Path
from typing import List, Dict

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


class KSDD2DenseDataset(Dataset):
    def __init__(self, root: str, image_size: int = 256, split: str = "test"):
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
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

        self.classes = ["ksdd2"]
        self.samples: List[Dict] = []
        self._build_index()

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {self.split_dir}")

    def _is_image(self, p: Path) -> bool:
        return p.is_file() and p.suffix in IMG_EXTS

    def _build_index(self):
        for img_path in sorted(self.split_dir.iterdir()):
            if not self._is_image(img_path):
                continue
            if img_path.stem.endswith("_GT"):
                continue

            mask_path = self.split_dir / f"{img_path.stem}_GT{img_path.suffix}"
            if not mask_path.exists():
                # Fallback for different GT extension.
                hits = sorted(self.split_dir.glob(f"{img_path.stem}_GT.*"))
                mask_path = hits[0] if hits else None

            label = 0
            if mask_path is not None:
                m = Image.open(mask_path).convert("L")
                label = 1 if m.getbbox() is not None else 0

            self.samples.append({
                "image_path": img_path,
                "mask_path": mask_path,
                "class_name": "ksdd2",
                "label": label,
                "defect_type": "defect" if label == 1 else "good",
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

        return {
            "image": image,
            "mask": mask,
            "label": torch.tensor(float(item["label"]), dtype=torch.float32),
            "class_name": item["class_name"],
            "defect_type": item["defect_type"],
            "image_path": str(item["image_path"]),
            "mask_path": str(item["mask_path"]) if item["mask_path"] is not None else "",
        }


if __name__ == "__main__":
    ds = KSDD2DenseDataset("../datasets_extra/ksdd2", image_size=256)
    print("num samples:", len(ds))
    print("classes:", ds.classes)
    x = ds[0]
    print(x["image"].shape, x["mask"].shape, x["label"], x["class_name"], x["defect_type"], x["image_path"])
