"""
Evaluate raw MobileCLIP2 stage-text similarity for dense anomaly localization.

This diagnostic evaluates whether raw MobileCLIP2 hierarchical stage features
are directly suitable for dense text matching, without any learnable projection,
local refinement, prompt learning, or prototype anchoring.

Main setting:
  - Backbone: MobileCLIP2-S3
  - Stage: 3 only
  - Raw feature dim: 768
  - Text dim: 768
  - Text prototypes: fixed "normal object" / "defective object"
  - Anomaly score: cos(stage3, defective) - cos(stage3, normal)
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

# Make project-root imports work when this script is executed as:
#   python scripts/eval_raw_mobileclip_stage_text_similarity.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score, average_precision_score

from datasets.mvtec_dataset import MVTecDenseDataset
from datasets.visa_dataset import VisADenseDataset
from datasets.btad_dataset import BTADDenseDataset
from datasets.mpdd_dataset import MPDDDenseDataset
from datasets.dagm_dataset import DAGMDenseDataset
from datasets.mvtec_loco_dataset import MVTecLOCODenseDataset
from datasets.ksdd2_dataset import KSDD2DenseDataset
from datasets.severstal_dataset import SeverstalDenseDataset

from models.mobileclip_extractor import MobileCLIPStageExtractor
from models.prompts import build_text_prototypes

# Reuse the exact metric helpers from the main eval script.
from scripts.eval_densealign_learnable_prompt import (
    topk_score,
    best_f1_iou_from_scores,
    compute_aupro,
)


def get_batch_item(batch, keys, default=None):
    for k in keys:
        if k in batch:
            return batch[k]
    return default



def topk_score_numpy(score_map, ratio=0.01):
    flat = score_map.reshape(-1)
    k = max(1, int(flat.size * ratio))
    idx = np.argpartition(flat, -k)[-k:]
    return float(flat[idx].mean())


def build_dataset(args):
    if args.eval_dataset == "mvtec":
        return MVTecDenseDataset(
            args.mvtec_root,
            image_size=args.image_size,
        )
    if args.eval_dataset == "visa":
        return VisADenseDataset(
            root=args.visa_root,
            split_csv=args.visa_split_csv,
            split=args.visa_split,
            image_size=args.image_size,
        )
    if args.eval_dataset == "btad":
        return BTADDenseDataset(
            root=args.btad_root,
            split="test",
            image_size=args.image_size,
        )
    if args.eval_dataset == "mpdd":
        return MPDDDenseDataset(
            root=args.mpdd_root,
            split="test",
            image_size=args.image_size,
        )
    if args.eval_dataset == "dagm":
        return DAGMDenseDataset(
            root=args.dagm_root,
            image_size=args.image_size,
            split="Test",
        )
    if args.eval_dataset == "mvtec_loco":
        return MVTecLOCODenseDataset(
            root=args.mvtec_loco_root,
            split="test",
            image_size=args.image_size,
        )
    if args.eval_dataset == "ksdd2":
        return KSDD2DenseDataset(
            root=args.ksdd2_root,
            split="test",
            image_size=args.image_size,
        )
    if args.eval_dataset == "severstal":
        return SeverstalDenseDataset(
            root=args.severstal_root,
            split="test",
            image_size=args.image_size,
        )
    raise ValueError(f"Unknown eval_dataset: {args.eval_dataset}")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_dataset", type=str, default="mvtec",
                        choices=["mvtec", "visa", "btad", "mpdd", "dagm", "mvtec_loco", "ksdd2", "severstal"])

    parser.add_argument("--mvtec_root", type=str, default="../datasets/mvtec_anomaly_detection")
    parser.add_argument("--visa_root", type=str, default="../datasets/visa")
    parser.add_argument("--visa_split_csv", type=str, default="../datasets/visa/split_csv/1cls.csv")
    parser.add_argument("--visa_split", type=str, default="test")
    parser.add_argument("--btad_root", type=str, default="../datasets/BTAD")
    parser.add_argument("--mpdd_root", type=str, default="../datasets/MPDD")
    parser.add_argument("--dagm_root", type=str, default="../datasets/DAGM")
    parser.add_argument("--mvtec_loco_root", type=str, default="../datasets/mvtec_loco_anomaly_detection")
    parser.add_argument("--ksdd2_root", type=str, default="../datasets/KSDD2")
    parser.add_argument("--severstal_root", type=str, default="../datasets/severstal")

    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="MobileCLIP2-S3")
    parser.add_argument("--pretrained", type=str, default="dfndr2b")
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--normal_prompt", type=str, default="normal object")
    parser.add_argument("--abnormal_prompt", type=str, default="defective object")
    parser.add_argument(
        "--score_mode",
        type=str,
        default="margin",
        choices=["abnormal", "inverse_normal", "margin", "inverse_margin"],
        help="Raw dense score: abnormal=s_abn, inverse_normal=1-s_norm, margin=s_abn-s_norm, inverse_margin=s_norm-s_abn.",
    )

    parser.add_argument("--aupro_max_fpr", type=float, default=0.3)
    parser.add_argument("--aupro_thresholds", type=int, default=200)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("eval_dataset:", args.eval_dataset)
    print("model_name:", args.model_name)
    print("stage:", args.stage)
    print("normal_prompt:", args.normal_prompt)
    print("abnormal_prompt:", args.abnormal_prompt)
    print("score_mode:", args.score_mode)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args)
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
        stages=(args.stage,),
        device=device,
        freeze=True,
    ).to(device).eval()

    # Build fixed object-agnostic text prototypes.
    # build_text_prototypes expects class names, so use the same fixed phrase
    # for all classes through class_name="object" style if possible.
    # For raw diagnostic, directly tokenize explicit prompts using open_clip model.
    import open_clip

    tokenizer = open_clip.get_tokenizer(args.model_name)
    texts = tokenizer([args.normal_prompt, args.abnormal_prompt]).to(device)
    text_features = extractor.model.encode_text(texts)
    text_features = F.normalize(text_features, dim=-1)
    t_normal = text_features[0]
    t_abnormal = text_features[1]

    all_data = defaultdict(list)

    for batch in tqdm(loader, desc="raw-eval"):
        images = get_batch_item(batch, ["image", "img", "images"]).to(device)
        masks = get_batch_item(batch, ["mask", "gt_mask", "masks"])
        labels = get_batch_item(batch, ["label", "is_anomaly", "labels"])
        class_names = get_batch_item(batch, ["class_name", "cls_name", "category"])
        image_paths = get_batch_item(batch, ["image_path", "path", "img_path"], default=None)

        if masks is None:
            raise KeyError("Batch does not contain mask key.")
        if labels is None:
            raise KeyError("Batch does not contain label key.")
        if class_names is None:
            raise KeyError("Batch does not contain class_name key.")

        feats = extractor(images)[args.stage]  # [B, 768, h, w]
        feats = F.normalize(feats, dim=1)

        sim_n = torch.einsum("bchw,c->bhw", feats, t_normal)
        sim_a = torch.einsum("bchw,c->bhw", feats, t_abnormal)

        if args.score_mode == "abnormal":
            logits = sim_a
        elif args.score_mode == "inverse_normal":
            logits = 1.0 - sim_n
        elif args.score_mode == "margin":
            logits = sim_a - sim_n
        elif args.score_mode == "inverse_margin":
            logits = sim_n - sim_a
        else:
            raise ValueError(f"Unknown score_mode: {args.score_mode}")

        logits = F.interpolate(
            logits.unsqueeze(1),
            size=(args.image_size, args.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        scores = logits.detach().cpu().float().numpy()
        masks_np = masks.detach().cpu().float().numpy()
        labels_np = labels.detach().cpu().numpy()

        if masks_np.ndim == 4:
            masks_np = masks_np[:, 0]
        masks_np = (masks_np > 0.5).astype(np.uint8)

        # image-level scores
        score_max = scores.reshape(scores.shape[0], -1).max(axis=1)
        score_mean = scores.reshape(scores.shape[0], -1).mean(axis=1)

        score_top1 = np.asarray([topk_score_numpy(s, ratio=0.01) for s in scores])
        score_top5 = np.asarray([topk_score_numpy(s, ratio=0.05) for s in scores])
        score_top10 = np.asarray([topk_score_numpy(s, ratio=0.10) for s in scores])

        for i in range(scores.shape[0]):
            cn = class_names[i] if isinstance(class_names, (list, tuple)) else class_names[i]
            if not isinstance(cn, str):
                cn = str(cn)

            path = ""
            if image_paths is not None:
                path = image_paths[i] if isinstance(image_paths, (list, tuple)) else str(image_paths[i])

            all_data[cn].append({
                "score_map": scores[i],
                "mask": masks_np[i],
                "label": int(labels_np[i]),
                "image_path": path,
                "score_max": float(score_max[i]),
                "score_top1": float(score_top1[i]),
                "score_top5": float(score_top5[i]),
                "score_top10": float(score_top10[i]),
                "score_mean": float(score_mean[i]),
            })

    score_names = ["score_max", "score_top1", "score_top5", "score_top10", "score_mean"]

    class_rows = []

    for class_name, items in all_data.items():
        masks = np.stack([x["mask"] for x in items], axis=0)
        maps = np.stack([x["score_map"] for x in items], axis=0)
        labels = np.asarray([x["label"] for x in items], dtype=np.int32)

        pix_y = masks.reshape(-1)
        pix_s = maps.reshape(-1)

        pixel_auroc = roc_auc_score(pix_y, pix_s) if len(np.unique(pix_y)) > 1 else np.nan
        pixel_ap = average_precision_score(pix_y, pix_s) if len(np.unique(pix_y)) > 1 else np.nan
        pixel_f1, pixel_iou = best_f1_iou_from_scores(pix_y, pix_s)
        pixel_aupro = compute_aupro(
            maps,
            masks,
            max_fpr=args.aupro_max_fpr,
            num_thresholds=args.aupro_thresholds,
        )

        for score_name in score_names:
            img_s = np.asarray([x[score_name] for x in items], dtype=np.float32)
            img_y = labels

            image_auroc = roc_auc_score(img_y, img_s) if len(np.unique(img_y)) > 1 else np.nan
            image_ap = average_precision_score(img_y, img_s) if len(np.unique(img_y)) > 1 else np.nan
            image_f1, _ = best_f1_iou_from_scores(img_y, img_s)

            class_rows.append({
                "class_name": class_name,
                "score_name": score_name,
                "pixel_auroc": pixel_auroc,
                "pixel_ap": pixel_ap,
                "pixel_f1": pixel_f1,
                "pixel_iou": pixel_iou,
                "pixel_aupro": pixel_aupro,
                "image_auroc": image_auroc,
                "image_ap": image_ap,
                "image_f1": image_f1,
            })

    class_df = pd.DataFrame(class_rows)
    class_csv = output_dir / f"{args.eval_dataset}_classwise_metrics.csv"
    class_df.to_csv(class_csv, index=False)
    print("saved:", class_csv)

    per_image_rows = []
    for class_name, items in all_data.items():
        for x in items:
            row = {
                "class_name": class_name,
                "image_path": x["image_path"],
                "label": x["label"],
            }
            for score_name in score_names:
                row[score_name] = x[score_name]
            per_image_rows.append(row)

    per_image_csv = output_dir / f"{args.eval_dataset}_per_image_scores.csv"
    pd.DataFrame(per_image_rows).to_csv(per_image_csv, index=False)
    print("saved:", per_image_csv)

    summary_rows = []
    for score_name in score_names:
        sub = class_df[class_df["score_name"] == score_name]
        summary_rows.append({
            "score_name": score_name,
            "macro_pixel_auroc": sub["pixel_auroc"].mean(),
            "macro_pixel_ap": sub["pixel_ap"].mean(),
            "macro_pixel_f1": sub["pixel_f1"].mean(),
            "macro_pixel_iou": sub["pixel_iou"].mean(),
            "macro_pixel_aupro": sub["pixel_aupro"].mean(),
            "macro_image_auroc": sub["image_auroc"].mean(),
            "macro_image_ap": sub["image_ap"].mean(),
            "macro_image_f1": sub["image_f1"].mean(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / f"{args.eval_dataset}_summary_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n===== SUMMARY =====")
    print(summary_df.to_string(index=False))
    print("\nsaved:", summary_csv)


if __name__ == "__main__":
    main()
