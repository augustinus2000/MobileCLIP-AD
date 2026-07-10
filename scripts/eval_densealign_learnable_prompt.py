"""
Evaluate MobileCLIP-AD / LDA checkpoints.

Verified evaluation setting:
  - Use best.pth for evaluation.
  - If --align_mode is not specified, the evaluator uses the align_mode stored
    in the checkpoint. Final MobileCLIP-AD checkpoints use local LDA.
  - For VisA evaluation, use --visa_eval_split test.
  - For MVTec evaluation, use the official MVTec AD test set.

Verified command: VisA-trained S3 -> MVTec

  python scripts/eval_densealign_learnable_prompt.py \
    --eval_dataset mvtec \
    --mvtec_root ../datasets/mvtec_anomaly_detection \
    --ckpt outputs/lda_s3_objprompt_objanchor_seed0_visa/best.pth \
    --output_dir paper_results/source_csv/lda_visa_to_mvtec_s3_seed0 \
    --batch_size 32 \
    --num_workers 4 \
    --aupro_thresholds 200

Verified command: MVTec-trained S3 -> VisA

  python scripts/eval_densealign_learnable_prompt.py \
    --eval_dataset visa \
    --visa_root ../datasets/visa \
    --visa_split_csv ../datasets/visa/split_csv/1cls.csv \
    --visa_eval_split test \
    --ckpt outputs/lda_s3_objprompt_objanchor_seed0_mvtec/best.pth \
    --output_dir paper_results/source_csv/lda_mvtec_to_visa_s3_seed0 \
    --batch_size 32 \
    --num_workers 4 \
    --aupro_thresholds 200

Verified command: ViT-L/14-336 reference, VisA-trained -> MVTec

  python scripts/eval_densealign_learnable_prompt.py \
    --eval_dataset mvtec \
    --mvtec_root ../datasets/mvtec_anomaly_detection \
    --ckpt outputs/lda_vitl14_336_objprompt_objanchor_seed0_visa/best.pth \
    --output_dir paper_results/source_csv/lda_vitl14_336_visa_to_mvtec_seed0 \
    --batch_size 32 \
    --num_workers 4 \
    --aupro_thresholds 200

For additional full-run commands, see the top-level run_*.sh files.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import open_clip
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

from datasets.mvtec_dataset import MVTecDenseDataset
from datasets.visa_dataset import VisADenseDataset
from datasets.btad_dataset import BTADDenseDataset
from datasets.mpdd_dataset import MPDDDenseDataset
from datasets.dagm_dataset import DAGMDenseDataset
from datasets.mvtec_loco_dataset import MVTecLOCODenseDataset
from datasets.ksdd2_dataset import KSDD2DenseDataset
from datasets.severstal_dataset import SeverstalDenseDataset
from models.mobileclip_extractor import MobileCLIPStageExtractor
from models.clip_vit_extractor import CLIPViTStageExtractor
from models.densealign import DenseAlignHead
from models.prompts import build_text_prototypes
from models.learnable_prompt import ClassAgnosticLearnablePrompt


def topk_score(logits, ratio=0.01):
    """
    logits: [B,1,H,W]
    """
    b = logits.shape[0]
    flat = logits.view(b, -1)
    k = max(1, int(flat.shape[1] * ratio))
    return flat.topk(k, dim=1).values.mean(dim=1)


def best_f1_iou_from_scores(y_true, y_score):
    """
    Pixel-level best F1 / IoU over PR thresholds.
    """
    y_true = y_true.astype(np.uint8)
    y_score = y_score.astype(np.float32)

    if y_true.max() == 0:
        return np.nan, np.nan

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    # precision/recall length = len(thresholds)+1
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.nanargmax(f1))
    best_f1 = float(f1[best_idx])

    if best_idx >= len(thresholds):
        thr = thresholds[-1]
    else:
        thr = thresholds[best_idx]

    pred = y_score >= thr
    gt = y_true > 0

    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = float(inter / (union + 1e-12))

    return best_f1, iou


def connected_components(mask):
    """
    Return list of boolean masks, one per connected GT region.
    Uses scipy if available, with a small fallback otherwise.
    """
    mask = mask.astype(bool)

    try:
        from scipy import ndimage
        labeled, n = ndimage.label(mask)
        return [(labeled == i) for i in range(1, n + 1)]
    except Exception:
        # fallback: simple BFS 4-connectivity
        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        comps = []

        for y in range(h):
            for x in range(w):
                if not mask[y, x] or visited[y, x]:
                    continue

                stack = [(y, x)]
                visited[y, x] = True
                coords = []

                while stack:
                    cy, cx = stack.pop()
                    coords.append((cy, cx))

                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < h and 0 <= nx < w:
                            if mask[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                stack.append((ny, nx))

                comp = np.zeros_like(mask, dtype=bool)
                yy, xx = zip(*coords)
                comp[np.array(yy), np.array(xx)] = True
                comps.append(comp)

        return comps


def compute_aupro(scores, masks, max_fpr=0.3, num_thresholds=200):
    """
    scores: list of [H,W] anomaly score maps, higher = more anomalous
    masks : list of [H,W] binary GT masks

    AUPRO:
    - For each threshold, compute FPR over non-defect pixels.
    - Compute per-region overlap for each connected GT component.
    - Average PRO over all GT regions.
    - Integrate PRO over FPR in [0, max_fpr], normalized by max_fpr.
    """
    assert len(scores) == len(masks)

    scores = [np.asarray(s, dtype=np.float32) for s in scores]
    masks = [np.asarray(m, dtype=np.uint8) > 0 for m in masks]

    all_scores = np.concatenate([s.reshape(-1) for s in scores])
    if all_scores.size == 0:
        return np.nan

    # If all maps are constant, AUPRO is not meaningful.
    s_min, s_max = float(all_scores.min()), float(all_scores.max())
    if abs(s_max - s_min) < 1e-12:
        return np.nan

    thresholds = np.linspace(s_max, s_min, num_thresholds)

    regions = []
    normal_pixel_total = 0

    for m in masks:
        comps = connected_components(m)
        regions.extend(comps)
        normal_pixel_total += int((~m).sum())

    if len(regions) == 0 or normal_pixel_total == 0:
        return np.nan

    fprs = []
    pros = []

    for thr in thresholds:
        false_positive = 0
        region_overlaps = []

        for score_map, gt in zip(scores, masks):
            pred = score_map >= thr

            false_positive += int(np.logical_and(pred, ~gt).sum())

            comps = connected_components(gt)
            for comp in comps:
                denom = int(comp.sum())
                if denom > 0:
                    overlap = np.logical_and(pred, comp).sum() / denom
                    region_overlaps.append(float(overlap))

        fpr = false_positive / (normal_pixel_total + 1e-12)
        pro = float(np.mean(region_overlaps)) if len(region_overlaps) > 0 else np.nan

        fprs.append(fpr)
        pros.append(pro)

    fprs = np.asarray(fprs, dtype=np.float64)
    pros = np.asarray(pros, dtype=np.float64)

    valid = np.isfinite(fprs) & np.isfinite(pros)
    fprs = fprs[valid]
    pros = pros[valid]

    if len(fprs) < 2:
        return np.nan

    # Sort by FPR ascending.
    order = np.argsort(fprs)
    fprs = fprs[order]
    pros = pros[order]

    # Keep only FPR <= max_fpr.
    keep = fprs <= max_fpr
    fprs_keep = fprs[keep]
    pros_keep = pros[keep]

    if len(fprs_keep) < 2:
        return np.nan

    # Ensure curve starts at 0.
    if fprs_keep[0] > 0:
        fprs_keep = np.concatenate([[0.0], fprs_keep])
        pros_keep = np.concatenate([[pros_keep[0]], pros_keep])

    # Ensure curve reaches max_fpr by linear interpolation.
    if fprs_keep[-1] < max_fpr:
        fprs_keep = np.concatenate([fprs_keep, [max_fpr]])
        pros_keep = np.concatenate([pros_keep, [pros_keep[-1]]])

    area = np.trapz(pros_keep, fprs_keep)
    return float(area / max_fpr)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_dataset", type=str, default="mvtec",
                        choices=["mvtec", "visa", "btad", "mpdd", "dagm", "mvtec_loco", "ksdd2", "severstal"],
                        help="Target dataset for evaluation.")
    parser.add_argument("--mvtec_root", type=str, default="../datasets/mvtec_anomaly_detection")
    parser.add_argument("--visa_root", type=str, default="../datasets/visa")
    parser.add_argument("--btad_root", type=str, default="../datasets/btad/BTech_Dataset_transformed")
    parser.add_argument("--mpdd_root", type=str, default="../datasets/mpdd_data/MPDD")
    parser.add_argument("--dagm_root", type=str, default="../datasets/dagm/DAGM2007/DAGM_KaggleUpload")
    parser.add_argument("--mvtec_loco_root", type=str, default="../datasets_extra/mvtec_loco_ad")
    parser.add_argument("--ksdd2_root", type=str, default="../datasets_extra/ksdd2")
    parser.add_argument("--ksdd2_split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--severstal_root", type=str, default="../datasets_extra/severstal")
    parser.add_argument("--visa_split_csv", type=str, default="../datasets/visa/split_csv/1cls.csv")
    parser.add_argument("--visa_eval_split", type=str, default="test",
                        choices=["train", "test", "all", "full"])
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="MobileCLIP2-S3")
    parser.add_argument("--pretrained", type=str, default="dfndr2b")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--align_mode", type=str, default=None,
                        choices=["projection", "local"],
                        help="Override alignment mode. If None, use checkpoint value. Final MobileCLIP-AD uses local/LDA.")
    parser.add_argument("--fixed_stage_fusion", action="store_true",
                        help="Override to use fixed equal stage fusion. If not set, use checkpoint value when available.")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--topk_ratio", type=float, default=0.01)
    parser.add_argument("--aupro_max_fpr", type=float, default=0.3)
    parser.add_argument("--aupro_thresholds", type=int, default=200)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("device:", device)
    print("ckpt:", args.ckpt)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    stage_dims = ckpt.get("stage_dims", {2: 384, 3: 768})
    stage_dims = {int(k): int(v) for k, v in stage_dims.items()}
    stages = tuple(sorted(stage_dims.keys()))

    print("stages:", stages)
    print("stage_dims:", stage_dims)

    # Load text embedding dimension from checkpoint if available.
    # Fallback is determined by MobileCLIP2 variant.
    mobileclip2_text_dims = {
        "MobileCLIP2-S0": 512,
        "MobileCLIP2-S2": 512,
        "MobileCLIP2-S3": 768,
        "MobileCLIP2-S4": 768,
    }
    model_name = ckpt.get("model_name", args.model_name)
    args.model_name = model_name  # ensure extractor/tokenizer/prompt use the checkpoint backbone
    text_dim = ckpt.get("text_dim", mobileclip2_text_dims.get(model_name, 768))
    print("model_name:", model_name)
    print("text_dim:", text_dim)

    print(f"[INFO] eval_dataset = {args.eval_dataset}")

    if args.eval_dataset == "mvtec":
        dataset = MVTecDenseDataset(args.mvtec_root, image_size=args.image_size)
    elif args.eval_dataset == "visa":
        dataset = VisADenseDataset(
            root=args.visa_root,
            image_size=args.image_size,
            split=args.visa_eval_split,
            split_csv=args.visa_split_csv,
        )
    elif args.eval_dataset == "btad":
        dataset = BTADDenseDataset(
            root=args.btad_root,
            image_size=args.image_size,
        )
    elif args.eval_dataset == "mpdd":
        dataset = MPDDDenseDataset(
            root=args.mpdd_root,
            image_size=args.image_size,
        )
    elif args.eval_dataset == "dagm":
        dataset = DAGMDenseDataset(
            root=args.dagm_root,
            image_size=args.image_size,
            split="Test",
        )
    elif args.eval_dataset == "mvtec_loco":
        dataset = MVTecLOCODenseDataset(
            root=args.mvtec_loco_root,
            image_size=args.image_size,
        )
    elif args.eval_dataset == "ksdd2":
        dataset = KSDD2DenseDataset(
            root=args.ksdd2_root,
            image_size=args.image_size,
            split=args.ksdd2_split,
        )
    elif args.eval_dataset == "severstal":
        dataset = SeverstalDenseDataset(
            root=args.severstal_root,
            image_size=args.image_size,
        )
    else:
        raise ValueError(f"Unknown eval_dataset: {args.eval_dataset}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    backbone_type = ckpt.get("backbone_type", "mobileclip")

    if backbone_type == "clip_vit":
        clip_model, _, _ = open_clip.create_model_and_transforms(
            args.model_name,
            pretrained=args.pretrained,
        )
        clip_model = clip_model.to(device).eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        extractor = CLIPViTStageExtractor(
            clip_model,
            layers=(12, 24),
        ).to(device)
        extractor.model = clip_model
    else:
        extractor = MobileCLIPStageExtractor(
            model_name=args.model_name,
            pretrained=args.pretrained,
            stages=stages,
            device=device,
            freeze=True,
        ).to(device)

    extractor.eval()

    ckpt_align_mode = ckpt.get("align_mode", "projection")
    align_mode = args.align_mode if args.align_mode is not None else ckpt_align_mode
    fixed_stage_fusion = bool(args.fixed_stage_fusion or ckpt.get("fixed_stage_fusion", False))
    print("align_mode:", align_mode)
    print("fixed_stage_fusion:", fixed_stage_fusion)

    head = DenseAlignHead(
        stage_dims=stage_dims,
        text_dim=text_dim,
        out_size=args.image_size,
        temperature_init=10.0,
        align_mode=align_mode,
        fixed_stage_fusion=fixed_stage_fusion,
    ).to(device)
    head.load_state_dict(ckpt["head"], strict=True)

    head.eval()

    tokenizer = open_clip.get_tokenizer(args.model_name)
    prototypes = build_text_prototypes(
        model=extractor.model,
        tokenizer=tokenizer,
        class_names=dataset.classes,
        device=device,
    )

    learnable_prompt = None
    if ckpt.get("learnable_prompt", None) is not None:
        n_ctx = int(ckpt.get("n_ctx", 4))
        learnable_prompt = ClassAgnosticLearnablePrompt(
            text_tower=(extractor.model.text if hasattr(extractor.model, 'text') else extractor.model),
            tokenizer=tokenizer,
            class_names=list(prototypes.keys()),
            n_ctx=n_ctx,
            normal_word="normal",
            abnormal_word="defective",
            object_agnostic=bool(ckpt.get("object_agnostic_prompt", False)),
            object_word=ckpt.get("object_word", "object"),
        ).to(device)

        lp_state = ckpt["learnable_prompt"]
        with torch.no_grad():
            learnable_prompt.normal_ctx.copy_(lp_state["normal_ctx"].to(device))
            learnable_prompt.abnormal_ctx.copy_(lp_state["abnormal_ctx"].to(device))

        learnable_prompt.eval()
        prototypes = learnable_prompt(list(prototypes.keys()))
        print(f"loaded learnable_prompt: n_ctx={n_ctx}")

    per_class = {
        c: {
            "pixel_scores": [],
            "pixel_gts": [],
            "image_scores_max": [],
            "image_scores_top1": [],
            "image_scores_top5": [],
            "image_scores_top10": [],
            "image_scores_mean": [],
            "image_gts": [],
            "image_paths": [],
            "defect_types": [],
            "image_dice": [],
            "image_iou": [],
            "image_pred_area": [],
            "image_gt_area": [],
            "maps": [],
            "masks": [],
        }
        for c in dataset.classes
    }

    for batch in tqdm(loader, desc="eval"):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        class_names = batch["class_name"]

        feats = extractor(images)

        logits_list = []
        for i, cls in enumerate(class_names):
            n, a = prototypes[cls]
            out = head(
                feats,
                n.unsqueeze(0).to(device),
                a.unsqueeze(0).to(device),
            )
            logits_list.append(out["logits"][i:i+1])

        logits = torch.cat(logits_list, dim=0)

        b = logits.shape[0]
        flat = logits.view(b, -1)

        score_max = flat.max(dim=1).values
        score_top1 = topk_score(logits, ratio=0.01)
        score_top5 = topk_score(logits, ratio=0.05)
        score_top10 = topk_score(logits, ratio=0.10)
        score_mean = flat.mean(dim=1)

        logits_np = logits[:, 0].detach().cpu().numpy()
        masks_np = masks[:, 0].detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        image_paths = batch.get("image_path", [""] * b)
        defect_types = batch.get("defect_type", [""] * b)

        for i, cls in enumerate(class_names):
            d = per_class[cls]

            d["pixel_scores"].append(logits_np[i].reshape(-1))
            d["pixel_gts"].append((masks_np[i].reshape(-1) > 0).astype(np.uint8))

            d["maps"].append(logits_np[i].astype(np.float32))
            d["masks"].append((masks_np[i] > 0).astype(np.uint8))

            d["image_scores_max"].append(float(score_max[i].detach().cpu()))
            d["image_scores_top1"].append(float(score_top1[i].detach().cpu()))
            d["image_scores_top5"].append(float(score_top5[i].detach().cpu()))
            d["image_scores_top10"].append(float(score_top10[i].detach().cpu()))
            d["image_scores_mean"].append(float(score_mean[i].detach().cpu()))
            d["image_gts"].append(int(labels_np[i]))
            d["image_paths"].append(image_paths[i])
            d["defect_types"].append(defect_types[i])

            gt = (masks_np[i] > 0).astype(np.uint8)
            pred = (logits_np[i] > 0).astype(np.uint8)
            inter = float((pred & gt).sum())
            pred_sum = float(pred.sum())
            gt_sum = float(gt.sum())
            dice = (2.0 * inter) / (pred_sum + gt_sum + 1e-12)
            iou = inter / (pred_sum + gt_sum - inter + 1e-12)

            d["image_dice"].append(dice)
            d["image_iou"].append(iou)
            d["image_pred_area"].append(pred_sum / pred.size)
            d["image_gt_area"].append(gt_sum / gt.size)

    score_names = ["score_max", "score_top1", "score_top5", "score_top10", "score_mean"]

    class_rows = []

    for cls in dataset.classes:
        d = per_class[cls]

        pix_y = np.concatenate(d["pixel_gts"])
        pix_s = np.concatenate(d["pixel_scores"])

        pixel_auroc = roc_auc_score(pix_y, pix_s) if len(np.unique(pix_y)) > 1 else np.nan
        pixel_ap = average_precision_score(pix_y, pix_s) if len(np.unique(pix_y)) > 1 else np.nan
        pixel_f1, pixel_iou = best_f1_iou_from_scores(pix_y, pix_s)
        pixel_aupro = compute_aupro(
            d["maps"],
            d["masks"],
            max_fpr=args.aupro_max_fpr,
            num_thresholds=args.aupro_thresholds,
        )

        img_y = np.asarray(d["image_gts"], dtype=np.uint8)

        for score_name in score_names:
            img_s = np.asarray(d[f"image_scores_{score_name.replace('score_', '')}"], dtype=np.float32)

            image_auroc = roc_auc_score(img_y, img_s) if len(np.unique(img_y)) > 1 else np.nan
            image_ap = average_precision_score(img_y, img_s) if len(np.unique(img_y)) > 1 else np.nan

            # image F1
            if len(np.unique(img_y)) > 1:
                precision, recall, thresholds = precision_recall_curve(img_y, img_s)
                f1 = 2 * precision * recall / (precision + recall + 1e-12)
                image_f1 = float(np.nanmax(f1))
            else:
                image_f1 = np.nan

            class_rows.append({
                "class_name": cls,
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

    per_image_rows = []
    for cls, d in per_class.items():
        n = len(d["image_gts"])
        for i in range(n):
            per_image_rows.append({
                "class_name": cls,
                "defect_type": d["defect_types"][i],
                "image_path": d["image_paths"][i],
                "label": d["image_gts"][i],
                "score_max": d["image_scores_max"][i],
                "score_top1": d["image_scores_top1"][i],
                "score_top5": d["image_scores_top5"][i],
                "score_top10": d["image_scores_top10"][i],
                "score_mean": d["image_scores_mean"][i],
                "dice": d["image_dice"][i],
                "iou": d["image_iou"][i],
                "pred_area": d["image_pred_area"][i],
                "gt_area": d["image_gt_area"][i],
            })
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
    print("\nsaved:", class_csv)
    print("saved:", summary_csv)


if __name__ == "__main__":
    main()
