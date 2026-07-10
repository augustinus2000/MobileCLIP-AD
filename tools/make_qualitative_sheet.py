"""
Generate random qualitative audit sheets.

This utility samples MVTec examples and creates contact sheets to visually
inspect anomaly localization behavior across model variants.
"""

# Example usage:
#   python tools/make_qualitative_sheet.py
#
# This utility may require editing checkpoint/output paths inside the file
# depending on which variants should be compared.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import open_clip

from datasets.mvtec_dataset import MVTecDenseDataset
from models.mobileclip_extractor import MobileCLIPStageExtractor
from models.densealign import DenseAlignHead
from models.prompts import build_text_prototypes


CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)


def denorm_image(x):
    arr = x.detach().cpu().numpy()
    arr = arr * CLIP_STD + CLIP_MEAN
    arr = np.clip(arr, 0, 1)
    return np.transpose(arr, (1, 2, 0))


def normalize_map(m):
    m = np.asarray(m, dtype=np.float32)
    mn = float(m.min())
    mx = float(m.max())
    if mx - mn < 1e-8:
        return np.zeros_like(m)
    return (m - mn) / (mx - mn)


def select_random_indices(dataset, classes=None, samples_per_class=3, seed=0, only_anomaly=True):
    rng = random.Random(seed)

    buckets = {}
    for i, s in enumerate(dataset.samples):
        cls = s["class_name"]
        label = int(s["label"])

        if classes is not None and cls not in classes:
            continue
        if only_anomaly and label == 0:
            continue

        buckets.setdefault(cls, []).append(i)

    selected = []
    selected_info = []

    for cls in sorted(buckets.keys()):
        inds = buckets[cls]
        rng.shuffle(inds)
        chosen = inds[:samples_per_class]
        selected.extend(chosen)
        for idx in chosen:
            selected_info.append((idx, cls, dataset.samples[idx]["defect_type"], dataset.samples[idx]["image_path"]))

    return selected, selected_info


def load_head(ckpt_path, image_size, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    stage_dims = ckpt.get("stage_dims", {2: 384, 3: 768})
    stage_dims = {int(k): int(v) for k, v in stage_dims.items()}
    stages = tuple(sorted(stage_dims.keys()))

    head = DenseAlignHead(
        stage_dims=stage_dims,
        text_dim=768,
        out_size=image_size,
        temperature_init=10.0,
    ).to(device)

    head.load_state_dict(ckpt["head"], strict=True)
    head.eval()

    return head, stages, stage_dims


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mvtec_root", type=str, default="../datasets/mvtec_anomaly_detection")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--ckpt_s3", type=str, required=True)
    parser.add_argument("--ckpt_s23", type=str, required=True)
    parser.add_argument("--ckpt_s234", type=str, required=True)
    parser.add_argument("--ckpt_final", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="MobileCLIP2-S3")
    parser.add_argument("--pretrained", type=str, default="dfndr2b")
    parser.add_argument("--image_size", type=int, default=256)

    parser.add_argument("--classes", type=str, nargs="*", default=None)
    parser.add_argument("--samples_per_class", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--overlay", action="store_true")
    parser.add_argument("--save_heatmap", action="store_true")
    parser.add_argument("--save_contact_sheet", action="store_true")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("===== Dataset =====")
    dataset = MVTecDenseDataset(args.mvtec_root, image_size=args.image_size)

    selected, selected_info = select_random_indices(
        dataset,
        classes=args.classes,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        only_anomaly=True,
    )

    print("selected:", len(selected))
    for x in selected_info[:20]:
        print(x)

    loader = DataLoader(
        Subset(dataset, selected),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("===== Load heads =====")
    ckpt_items = [
        ("S3", args.ckpt_s3),
        ("S23", args.ckpt_s23),
        ("S234", args.ckpt_s234),
        ("Final", args.ckpt_final),
    ]

    heads = {}
    all_stages = set()

    for name, path in ckpt_items:
        head, stages, stage_dims = load_head(path, args.image_size, device)
        heads[name] = {
            "head": head,
            "stages": stages,
            "stage_dims": stage_dims,
            "path": path,
        }
        all_stages.update(stages)
        print(name, "stages:", stages, "stage_dims:", stage_dims, "path:", path)

    all_stages = tuple(sorted(all_stages))
    print("extractor union stages:", all_stages)

    print("===== Extractor =====")
    extractor = MobileCLIPStageExtractor(
        model_name=args.model_name,
        pretrained=args.pretrained,
        stages=all_stages,
        device=device,
        freeze=True,
    ).to(device)
    extractor.eval()

    tokenizer = open_clip.get_tokenizer(args.model_name)
    prototypes = build_text_prototypes(
        model=extractor.model,
        tokenizer=tokenizer,
        class_names=dataset.classes,
        device=device,
    )

    contact_paths = []

    print("===== Visualize =====")

    for idx, batch in enumerate(loader):
        image = batch["image"].to(device)
        mask = batch["mask"][0, 0].detach().cpu().numpy()
        cls = batch["class_name"][0]
        defect = batch["defect_type"][0]
        img_path = Path(batch["image_path"][0])

        n, a = prototypes[cls]
        text_n = n.unsqueeze(0).to(device)
        text_a = a.unsqueeze(0).to(device)

        feats_all = extractor(image)

        img_np = denorm_image(image[0])
        gt = mask.astype(np.float32)

        maps = {}
        raw_stats = {}

        for name, obj in heads.items():
            stages = obj["stages"]
            feats = {s: feats_all[s] for s in stages}

            out = obj["head"](feats, text_n, text_a)
            pred = out["logits"][0, 0].detach().cpu().numpy()

            maps[name] = normalize_map(pred)
            raw_stats[name] = {
                "raw_min": float(pred.min()),
                "raw_max": float(pred.max()),
                "raw_mean": float(pred.mean()),
            }

        save_dir = out_dir / cls / defect
        save_dir.mkdir(parents=True, exist_ok=True)

        stem = f"{idx:04d}_{cls}_{defect}_{img_path.stem}"

        # Main comparison figure
        names = ["Image", "GT", "S3", "S23", "S234", "Final"]
        fig, axes = plt.subplots(1, len(names), figsize=(22, 4))

        axes[0].imshow(img_np)
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("GT")
        axes[1].axis("off")

        for ax, name in zip(axes[2:], ["S3", "S23", "S234", "Final"]):
            if args.overlay:
                ax.imshow(img_np)
                ax.imshow(maps[name], cmap="jet", alpha=0.45, vmin=0, vmax=1)
            else:
                ax.imshow(maps[name], cmap="jet", vmin=0, vmax=1)

            st = heads[name]["stages"]
            ax.set_title(f"{name} {st}")
            ax.axis("off")

        fig.suptitle(f"{cls} / {defect} / {img_path.name}", fontsize=11)
        plt.tight_layout()

        fig_path = save_dir / f"{stem}_audit_compare.png"
        plt.savefig(fig_path, dpi=160)
        plt.close(fig)

        contact_paths.append(fig_path)
        print("saved:", fig_path)

        if args.save_heatmap:
            for name in ["S3", "S23", "S234", "Final"]:
                fig2, ax2 = plt.subplots(1, 1, figsize=(4, 4))
                ax2.imshow(maps[name], cmap="jet", vmin=0, vmax=1)
                ax2.set_title(f"{name} | {cls}/{defect}")
                ax2.axis("off")
                plt.tight_layout()
                plt.savefig(save_dir / f"{stem}_{name}_heatmap.png", dpi=160)
                plt.close(fig2)

        # save raw stats
        with open(save_dir / f"{stem}_raw_stats.txt", "w") as f:
            f.write(f"class={cls}\n")
            f.write(f"defect={defect}\n")
            f.write(f"image_path={img_path}\n")
            f.write(f"gt_area={float(gt.mean())}\n")
            for name, st in raw_stats.items():
                f.write(f"{name}: {st}\n")

    if args.save_contact_sheet:
        try:
            from PIL import Image

            imgs = []
            for p in contact_paths:
                im = Image.open(p).convert("RGB")
                imgs.append(im)

            if len(imgs) > 0:
                # Resize contact rows to common width
                max_w = max(im.width for im in imgs)
                resized = []
                for im in imgs:
                    if im.width != max_w:
                        scale = max_w / im.width
                        new_h = int(im.height * scale)
                        im = im.resize((max_w, new_h))
                    resized.append(im)

                total_h = sum(im.height for im in resized)
                sheet = Image.new("RGB", (max_w, total_h), "white")

                y = 0
                for im in resized:
                    sheet.paste(im, (0, y))
                    y += im.height

                sheet_path = out_dir / "random_audit_contact_sheet.jpg"
                sheet.save(sheet_path, quality=95)
                print("saved contact sheet:", sheet_path)

        except Exception as e:
            print("failed contact sheet:", repr(e))

    print("done")


if __name__ == "__main__":
    main()
