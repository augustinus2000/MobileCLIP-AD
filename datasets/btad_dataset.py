
"""
BTAD / BTech dataset loader.

Structure:
root/
 ├── 01/
 │   ├── test/
 │   │   ├── ok/
 │   │   └── ko/
 │   └── ground_truth/
 │       └── ko/
"""

from pathlib import Path
from typing import List, Dict, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp",
            ".JPG", ".JPEG", ".PNG"}


class BTADDenseDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_size: int = 256,
        classes: Optional[List[str]] = None,
    ):
        self.root = Path(root)
        self.image_size = image_size

        self.image_tf = T.Compose([
            T.Resize((image_size, image_size),
                     interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

        self.mask_tf = T.Compose([
            T.Resize((image_size, image_size),
                     interpolation=T.InterpolationMode.NEAREST),
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


    def _is_image(self, p: Path):
        return p.is_file() and p.suffix in IMG_EXTS


    def _build_index(self):
        for cls in self.classes:
            cls_dir = self.root / cls

            test_root = cls_dir / "test"
            gt_root = cls_dir / "ground_truth"

            for defect_dir in sorted(test_root.iterdir()):
                if not defect_dir.is_dir():
                    continue

                defect_type = defect_dir.name
                is_normal = defect_type == "ok"

                for img_path in sorted(defect_dir.iterdir()):
                    if not self._is_image(img_path):
                        continue

                    if is_normal:
                        label = 0
                        mask_path = None
                    else:
                        label = 1
                        mask_path = gt_root / defect_type / f"{img_path.stem}.png"

                        if not mask_path.exists():
                            candidates = list(
                                (gt_root / defect_type).glob(f"{img_path.stem}*")
                            )
                            mask_path = candidates[0] if candidates else None

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
            mask = torch.zeros(
                1,
                self.image_size,
                self.image_size,
                dtype=torch.float32
            )
        else:
            m = Image.open(item["mask_path"]).convert("L")
            mask = self.mask_tf(m)
            mask = (mask > 0).float()

        return {
            "image": image,
            "mask": mask,
            "label": torch.tensor(float(item["label"])),
            "class_name": item["class_name"],
            "defect_type": item["defect_type"],
            "image_path": str(item["image_path"]),
            "mask_path": str(item["mask_path"]) if item["mask_path"] else "",
        }
