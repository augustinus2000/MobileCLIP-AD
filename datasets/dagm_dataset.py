from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class DAGMDenseDataset(Dataset):
    """
    DAGM 2007 dataset loader for evaluation.

    Expected root:
      ../datasets/dagm/DAGM2007/DAGM_KaggleUpload

    Structure:
      Class1/Test/0001.PNG
      Class1/Test/Label/0001_label.PNG

    If a label mask exists, the sample is anomalous.
    Otherwise, it is treated as normal with a zero mask.
    """

    def __init__(self, root, image_size=256, split="Test"):
        self.root = Path(root)
        self.image_size = image_size
        self.split = split

        if not self.root.exists():
            raise FileNotFoundError(f"DAGM root not found: {self.root}")

        self.samples = []

        class_dirs = sorted(
            [p for p in self.root.glob("Class*") if p.is_dir()],
            key=lambda p: int(p.name.replace("Class", ""))
        )

        for cls_dir in class_dirs:
            split_dir = cls_dir / split
            label_dir = split_dir / "Label"

            if not split_dir.exists():
                continue

            for img_path in sorted(split_dir.glob("*.PNG")):
                stem = img_path.stem
                mask_path = label_dir / f"{stem}_label.PNG"
                is_anomaly = mask_path.exists()

                self.samples.append({
                    "img_path": img_path,
                    "mask_path": mask_path if is_anomaly else None,
                    "class_name": cls_dir.name.lower(),
                    "is_anomaly": int(is_anomaly),
                    "anomaly_type": "defect" if is_anomaly else "good",
                })

        if len(self.samples) == 0:
            raise RuntimeError(f"No DAGM samples found under {self.root}")

        self.classes = sorted(
            list({x["class_name"] for x in self.samples}),
            key=lambda x: int(x.replace("class", ""))
        )

        self.img_tf = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        return self.img_tf(img)

    def _load_mask(self, mask_path):
        if mask_path is None:
            return torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)

        mask = Image.open(mask_path).convert("L")
        mask = TF.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=T.InterpolationMode.NEAREST,
        )
        mask = torch.from_numpy((np.array(mask) > 0).astype(np.float32)).unsqueeze(0)
        return mask

    def __getitem__(self, idx):
        item = self.samples[idx]

        image = self._load_image(item["img_path"])
        mask = self._load_mask(item["mask_path"])

        return {
            "image": image,
            "mask": mask,
            "label": torch.tensor(item["is_anomaly"], dtype=torch.long),
            "class_name": item["class_name"],
            "anomaly_type": item["anomaly_type"],
            "path": str(item["img_path"]),
        }
