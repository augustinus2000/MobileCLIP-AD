"""
Audit AUPRO threshold stability.

This utility recomputes AUPRO with different threshold resolutions to
verify that reported AUPRO values are numerically stable.
"""

# Example usage:
#   python tools/audit_aupro.py \
#     --eval_dir outputs/eval_densealign_s23_d5_sc \
#     --thresholds 200 500 1000
#
# Use this to verify that AUPRO is stable with respect to threshold count.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import open_clip
from scipy import ndimage

from datasets.mvtec_dataset import MVTecDenseDataset
from models.mobileclip_extractor import MobileCLIPStageExtractor
from models.densealign import DenseAlignHead
from models.prompts import build_text_prototypes


def fast_aupro(scores, masks, max_fpr=0.3, num_thresholds=200, desc="aupro"):
    scores = [np.asarray(s, dtype=np.float32) for s in scores]
    masks = [np.asarray(m, dtype=np.uint8) > 0 for m in masks]

    all_scores = np.concatenate([s.reshape(-1) for s in scores])
    s_min, s_max = float(all_scores.min()), float(all_scores.max())
    if abs(s_max - s_min) < 1e-12:
        return np.nan

    thresholds = np.linspace(s_max, s_min, num_thresholds)

    comps_per_img = []
    normal_pixel_total = 0
    total_regions = 0

    for m in masks:
        labeled, n = ndimage.label(m)
        comps = [(labeled == i) for i in range(1, n + 1)]
        comps_per_img.append(comps)
        total_regions += len(comps)
        normal_pixel_total += int((~m).sum())

    if total_regions == 0 or normal_pixel_total == 0:
        return np.nan

    fprs = []
    pros = []

    for thr in tqdm(thresholds, desc=desc, leave=False):
        fp = 0
        overlaps = []

        for score_map, gt, comps in zip(scores, masks, comps_per_img):
            pred = score_map >= thr
            fp += int(np.logical_and(pred, ~gt).sum())

            for comp in comps:
                denom = int(comp.sum())
                if denom > 0:
                    overlaps.append(float(np.logical_and(pred, comp).sum() / denom))

        fpr = fp / (normal_pixel_total + 1e-12)
        pro = float(np.mean(overlaps)) if overlaps else np.nan

        fprs.append(fpr)
        pros.append(pro)

    fprs = np.asarray(fprs, dtype=np.float64)
    pros = np.asarray(pros, dtype=np.float64)

    valid = np.isfinite(fprs) & np.isfinite(pros)
    fprs = fprs[valid]
    pros = pros[valid]

    if len(fprs) < 2:
        return np.nan

    order = np.argsort(fprs)
    fprs = fprs[order]
    pros = pros[order]

    keep = fprs <= max_fpr
    fprs_keep = fprs[keep]
    pros_keep = pros[keep]

    if len(fprs_keep) < 2:
        return np.nan

    if fprs_keep[0] > 0:
        fprs_keep = np.concatenate([[0.0], fprs_keep])
        pros_keep = np.concatenate([[pros_keep[0]], pros_keep])

    if fprs_keep[-1] < max_fpr:
        fprs_keep = np.concatenate([fprs_keep, [max_fpr]])
        pros_keep = np.concatenate([pros_keep, [pros_keep[-1]]])

    area = np.trapz(pros_keep, fprs_keep)
    return float(area / max_fpr)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvtec_root", type=str, default="../datasets/mvtec_anomaly_detection")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="MobileCLIP2-S3")
    parser.add_argument("--pretrained", type=str, default="dfndr2b")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_fpr", type=float, default=0.3)
    parser.add_argument("--thresholds", type=int, nargs="+", default=[200, 500, 1000])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("ckpt:", args.ckpt)
    print("thresholds:", args.thresholds)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    stage_dims = {int(k): int(v) for k, v in ckpt.get("stage_dims", {2: 384, 3: 768}).items()}
    stages = tuple(sorted(stage_dims.keys()))

    dataset = MVTecDenseDataset(args.mvtec_root, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    extractor = MobileCLIPStageExtractor(
        model_name=args.model_name,
        pretrained=args.pretrained,
        stages=stages,
        device=device,
        freeze=True,
    ).to(device).eval()

    head = DenseAlignHead(
        stage_dims=stage_dims,
        text_dim=768,
        out_size=args.image_size,
        temperature_init=10.0,
    ).to(device).eval()
    head.load_state_dict(ckpt["head"], strict=True)

    tokenizer = open_clip.get_tokenizer(args.model_name)
    prototypes = build_text_prototypes(
        model=extractor.model,
        tokenizer=tokenizer,
        class_names=dataset.classes,
        device=device,
    )

    per_class = {c: {"maps": [], "masks": []} for c in dataset.classes}

    for batch in tqdm(loader, desc="collect maps"):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        class_names = batch["class_name"]

        feats = extractor(images)

        logits_list = []
        for i, cls in enumerate(class_names):
            n, a = prototypes[cls]
            out = head(feats, n.unsqueeze(0).to(device), a.unsqueeze(0).to(device))
            logits_list.append(out["logits"][i:i+1])

        logits = torch.cat(logits_list, dim=0)
        logits_np = logits[:, 0].detach().cpu().numpy()
        masks_np = masks[:, 0].detach().cpu().numpy()

        for i, cls in enumerate(class_names):
            per_class[cls]["maps"].append(logits_np[i].astype(np.float32))
            per_class[cls]["masks"].append((masks_np[i] > 0).astype(np.uint8))

    rows = []

    for cls in tqdm(dataset.classes, desc="classes"):
        d = per_class[cls]
        row = {"class_name": cls}

        for t in args.thresholds:
            val = fast_aupro(
                d["maps"],
                d["masks"],
                max_fpr=args.max_fpr,
                num_thresholds=t,
                desc=f"{cls} T={t}",
            )
            row[f"aupro_{t}"] = val

        if 200 in args.thresholds and 1000 in args.thresholds:
            row["diff_200_1000"] = row["aupro_200"] - row["aupro_1000"]
        if 500 in args.thresholds and 1000 in args.thresholds:
            row["diff_500_1000"] = row["aupro_500"] - row["aupro_1000"]

        rows.append(row)
        print(row)

    df = pd.DataFrame(rows)

    macro = {"class_name": "MACRO"}
    for c in df.columns:
        if c != "class_name":
            macro[c] = df[c].mean()

    df = pd.concat([df, pd.DataFrame([macro])], ignore_index=True)

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("\n===== RESULT =====")
    print(df.to_string(index=False))
    print("saved:", out)


if __name__ == "__main__":
    main()
