"""
Train MobileCLIP-AD / LDA with object-agnostic learnable prompts.

Verified main setting:
  - Backbone: MobileCLIP2-S3
  - Stages: 2 3
  - Alignment: local LDA
  - Prompt: object-agnostic learnable prompt
  - Anchor: fixed "normal object" / "defective object" prototype anchor
  - Final anchor weight: 0.5
  - Checkpoint for evaluation: best.pth

Important:
  - For VisA source training, use --visa_train_split all.
    The default VisA train split contains only normal samples and is NOT the
    setting used for the paper's supervised cross-dataset experiments.

Verified reproduction command: VisA -> MVTec, seed 0

  python scripts/train_densealign_learnable_prompt.py \
    --train_dataset visa \
    --visa_root ../datasets/visa \
    --visa_split_csv ../datasets/visa/split_csv/1cls.csv \
    --visa_train_split all \
    --output_dir outputs/lda_s3_objprompt_objanchor_seed0_visa \
    --model_name MobileCLIP2-S3 \
    --pretrained dfndr2b \
    --align_mode local \
    --learnable_prompt \
    --object_agnostic_prompt \
    --object_agnostic_anchor \
    --n_ctx 4 \
    --lambda_prompt_anchor 0.5 \
    --lambda_prompt_reg 0.001 \
    --stages 2 3 \
    --image_size 256 \
    --batch_size 32 \
    --epochs 1 \
    --num_workers 4 \
    --seed 0 \
    --lambda_cons 0.05 \
    --mask_dilate 5 \
    --save_every 1

Evaluation command:

  python scripts/eval_densealign_learnable_prompt.py \
    --eval_dataset mvtec \
    --mvtec_root ../datasets/mvtec_anomaly_detection \
    --ckpt outputs/lda_s3_objprompt_objanchor_seed0_visa/best.pth \
    --output_dir paper_results/source_csv/lda_visa_to_mvtec_s3_seed0 \
    --batch_size 32 \
    --num_workers 4 \
    --aupro_thresholds 200

Verified reproduction command: MVTec -> VisA, seed 0

  python scripts/train_densealign_learnable_prompt.py \
    --train_dataset mvtec \
    --mvtec_root ../datasets/mvtec_anomaly_detection \
    --output_dir outputs/lda_s3_objprompt_objanchor_seed0_mvtec \
    --model_name MobileCLIP2-S3 \
    --pretrained dfndr2b \
    --align_mode local \
    --learnable_prompt \
    --object_agnostic_prompt \
    --object_agnostic_anchor \
    --n_ctx 4 \
    --lambda_prompt_anchor 0.5 \
    --lambda_prompt_reg 0.001 \
    --stages 2 3 \
    --image_size 256 \
    --batch_size 32 \
    --epochs 1 \
    --num_workers 4 \
    --seed 0 \
    --lambda_cons 0.05 \
    --mask_dilate 5 \
    --save_every 1

Verified ViT-L/14-336 reference training command: VisA -> MVTec, seed 0

  python scripts/train_densealign_learnable_prompt.py \
    --train_dataset visa \
    --visa_root ../datasets/visa \
    --visa_split_csv ../datasets/visa/split_csv/1cls.csv \
    --visa_train_split all \
    --output_dir outputs/lda_vitl14_336_objprompt_objanchor_seed0_visa \
    --backbone_type clip_vit \
    --model_name ViT-L-14-336 \
    --pretrained openai \
    --align_mode local \
    --learnable_prompt \
    --object_agnostic_prompt \
    --object_agnostic_anchor \
    --n_ctx 4 \
    --lambda_prompt_anchor 0.5 \
    --lambda_prompt_reg 0.001 \
    --stages 2 3 \
    --image_size 336 \
    --batch_size 32 \
    --epochs 1 \
    --num_workers 4 \
    --seed 0 \
    --lambda_cons 0.05 \
    --mask_dilate 5 \
    --save_every 1

For additional run scripts, see the top-level run_*.sh files.
"""

import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from tqdm import tqdm
import open_clip

from datasets.visa_dataset import VisADenseDataset
from datasets.mvtec_dataset import MVTecDenseDataset
from models.mobileclip_extractor import MobileCLIPStageExtractor
from models.clip_vit_extractor import CLIPViTStageExtractor
from models.densealign import DenseAlignHead
from models.prompts import build_text_prototypes
from models.learnable_prompt import ClassAgnosticLearnablePrompt


def dice_loss_with_logits(logits, targets, labels=None, eps=1e-6):
    """
    Dice loss for anomaly segmentation.

    Important:
    - If labels is provided, compute Dice only on anomalous images.
    - Normal images have empty masks, so including them in Dice often makes
      the loss stay near 1.0 and destabilizes localization learning.
    """
    probs = torch.sigmoid(logits)
    targets = targets.float()

    if labels is not None:
        keep = labels.float() > 0.5
        if keep.sum() == 0:
            # No anomaly sample in this batch. Skip dice.
            return logits.sum() * 0.0
        probs = probs[keep]
        targets = targets[keep]

    dims = (1, 2, 3)
    inter = (probs * targets).sum(dim=dims)
    union = probs.sum(dim=dims) + targets.sum(dim=dims)

    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def focal_loss_with_logits(logits, targets, alpha=0.75, gamma=2.0):
    """
    Pixel focal loss.

    alpha=0.75 gives more weight to anomaly pixels.
    With industrial AD masks, positive pixels are extremely sparse.
    """
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)

    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)

    loss = alpha_t * ((1.0 - pt) ** gamma) * bce
    return loss.mean()




class MaskDilateWrapper(Dataset):
    """
    Apply mask dilation to a dataset that returns a dict with key "mask".
    This is used for MVTec source training because the original MVTec
    evaluation dataset does not expose a mask_dilate argument.
    """
    def __init__(self, base_dataset, kernel_size=0):
        self.base_dataset = base_dataset
        self.kernel_size = int(kernel_size)

        # Expose common dataset attributes used by the training script.
        self.samples = getattr(base_dataset, "samples", None)
        self.classes = getattr(base_dataset, "classes", None)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        if self.kernel_size <= 1:
            return item

        item = dict(item)
        mask = item.get("mask", None)
        if mask is None:
            return item

        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask)

        original_dim = mask.dim()
        if original_dim == 2:
            mask4 = mask.float().unsqueeze(0).unsqueeze(0)
        elif original_dim == 3:
            mask4 = mask.float().unsqueeze(0)
        else:
            return item

        k = self.kernel_size
        pad = k // 2
        mask4 = F.max_pool2d(mask4, kernel_size=k, stride=1, padding=pad)

        if original_dim == 2:
            item["mask"] = mask4.squeeze(0).squeeze(0)
        else:
            item["mask"] = mask4.squeeze(0)

        return item


def set_seed(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # deterministic=True can slow down training and may break some ops,
    # so we keep benchmark disabled only.
    torch.backends.cudnn.benchmark = False


def dilate_binary_mask(mask, kernel_size: int):
    """
    mask: [B,1,H,W], binary 0/1
    kernel_size: odd int. If <=1, no dilation.
    """
    if kernel_size is None or kernel_size <= 1:
        return mask
    pad = kernel_size // 2
    return F.max_pool2d(mask.float(), kernel_size=kernel_size, stride=1, padding=pad)


def stage_consistency_loss(stage_logits, size=64):
    """
    Weak consistency among stage anomaly maps.

    stage_logits:
        dict stage_id -> [B,1,H,W], already upsampled to image size

    We compare low-resolution sigmoid maps so that high-frequency local details
    are not overly penalized.
    """
    if stage_logits is None or len(stage_logits) < 2:
        # Return zero tensor on the same device if possible.
        if stage_logits and len(stage_logits) > 0:
            v = next(iter(stage_logits.values()))
            return v.sum() * 0.0
        return torch.tensor(0.0)

    keys = sorted(stage_logits.keys())
    probs = []

    for k in keys:
        p = torch.sigmoid(stage_logits[k])
        p = F.interpolate(
            p,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )
        probs.append(p)

    loss = 0.0
    n = 0
    for i in range(len(probs)):
        for j in range(i + 1, len(probs)):
            loss = loss + F.mse_loss(probs[i], probs[j])
            n += 1

    return loss / max(n, 1)


def topk_image_score(logits, ratio=0.01):
    """
    logits: [B, 1, H, W]
    return: [B]
    """
    B = logits.shape[0]
    flat = logits.flatten(1)
    k = max(1, int(flat.shape[1] * ratio))
    score = flat.topk(k, dim=1).values.mean(dim=1)
    return score


def collate_text_prototypes(batch_class_names, prototypes, device):
    normal_list = []
    abnormal_list = []

    for cls in batch_class_names:
        n, a = prototypes[cls]
        normal_list.append(n)
        abnormal_list.append(a)

    text_normal = torch.stack(normal_list, dim=0).to(device)
    text_abnormal = torch.stack(abnormal_list, dim=0).to(device)

    return text_normal, text_abnormal



@torch.no_grad()
def build_object_agnostic_anchor_prototypes(model, tokenizer, device, object_word="object"):
    """Build fixed object-agnostic anchor prototypes from the frozen text encoder."""
    texts = [
        f"normal {object_word}",
        f"defective {object_word}",
    ]
    tokens = tokenizer(texts).to(device)

    try:
        feats = model.encode_text(tokens, normalize=True)
    except TypeError:
        feats = model.encode_text(tokens)
        feats = F.normalize(feats, dim=-1)

    feats = feats.float()
    feats = F.normalize(feats, dim=-1)
    return feats[0].detach(), feats[1].detach()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dataset", type=str, default="visa",
                        choices=["visa", "mvtec"],
                        help="Source dataset for training.")
    parser.add_argument("--visa_root", type=str, default="../datasets/visa")
    parser.add_argument("--mvtec_root", type=str, default="../datasets/mvtec_anomaly_detection")
    parser.add_argument("--visa_split_csv", type=str, default="../datasets/visa/split_csv/1cls.csv")
    parser.add_argument("--visa_train_split", type=str, default="train",
                        choices=["train", "test", "all", "full"])
    parser.add_argument("--output_dir", type=str, default="outputs/dense_align_visa_s23")
    parser.add_argument("--backbone_type", type=str, default="mobileclip", choices=["mobileclip","clip_vit"])
    parser.add_argument("--model_name", type=str, default="MobileCLIP2-S3")
    parser.add_argument("--pretrained", type=str, default="dfndr2b")

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--lambda_dice", type=float, default=1.0)
    parser.add_argument("--lambda_focal", type=float, default=1.0)
    parser.add_argument("--lambda_img", type=float, default=0.5)
    parser.add_argument("--topk_ratio", type=float, default=0.01)

    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--align_mode", type=str, default="projection",
                        choices=["projection", "local"],
                        help="Dense alignment mode. Use local for final LDA; projection/local are ablations.")
    parser.add_argument("--fixed_stage_fusion", action="store_true",
                        help="Use fixed equal stage fusion instead of learnable stage weights. Used for projection-only ablation.")
    parser.add_argument("--stages", type=int, nargs="+", default=[2, 3],
                        help="MobileCLIP stages to use, e.g. --stages 3 or --stages 2 3 or --stages 2 3 4")
    parser.add_argument("--mask_dilate", type=int, default=0,
                        help="Dilation kernel size for training masks. 0 disables dilation. Use odd values such as 3, 5, 7.")
    parser.add_argument("--lambda_cons", type=float, default=0.0,
                        help="Weight for stage consistency regularization.")
    parser.add_argument("--cons_size", type=int, default=64,
                        help="Resolution for stage consistency comparison.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility.")

    parser.add_argument("--learnable_prompt", action="store_true")
    parser.add_argument(
        "--object_agnostic_prompt",
        action="store_true",
        help="Replace class names with a generic object word in learnable prompts.",
    )
    parser.add_argument("--object_word", type=str, default="object")
    parser.add_argument(
        "--object_agnostic_anchor",
        action="store_true",
        help="Use fixed normal object / defective object prototypes as anchor targets.",
    )
    parser.add_argument("--n_ctx", type=int, default=4)
    parser.add_argument("--lambda_prompt_reg", type=float, default=0.001)
    parser.add_argument("--lambda_prompt_anchor", type=float, default=0.0)

    args = parser.parse_args()
    # Final MobileCLIP-AD recommended setting:
    #   --align_mode local --learnable_prompt --lambda_prompt_anchor 0.5
    # Projection/local modes are intended for ablation studies.

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stages = tuple(args.stages)
    print("using stages:", stages)

    # MobileCLIP2 variant-specific stage/text dimensions.
    # Verified at image_size=256 using forward hooks.
    mobileclip2_stage_dims = {
        "MobileCLIP2-S0": {2: 256, 3: 512},
        "MobileCLIP2-S2": {2: 320, 3: 640},
        "MobileCLIP2-S3": {1: 192, 2: 384, 3: 768, 4: 1536},
        "MobileCLIP2-S4": {2: 512, 3: 1024, 4: 2048},
    }
    mobileclip2_text_dims = {
        "MobileCLIP2-S0": 512,
        "MobileCLIP2-S2": 512,
        "MobileCLIP2-S3": 768,
        "MobileCLIP2-S4": 768,
    }

    if args.backbone_type == "clip_vit":
        stage_dims = {2: 1024, 3: 1024}
        text_dim = 768
    else:
        if args.model_name not in mobileclip2_stage_dims:
            raise ValueError(
                f"Unsupported model_name: {args.model_name}. "
                f"Supported: {sorted(mobileclip2_stage_dims.keys())}"
            )

        stage_dim_table = mobileclip2_stage_dims[args.model_name]
        missing_stages = [s for s in stages if s not in stage_dim_table]
        if missing_stages:
            raise ValueError(
                f"{args.model_name} does not provide stages {missing_stages}. "
                f"Available stages: {sorted(stage_dim_table.keys())}"
            )

        stage_dims = {s: stage_dim_table[s] for s in stages}
        text_dim = mobileclip2_text_dims[args.model_name]

    print("stage dims:", stage_dims)
    print("text dim:", text_dim)

    print("===== Dataset =====")
    print(f"[INFO] train_dataset = {args.train_dataset}")

    if args.train_dataset == "visa":
        base_train_ds = VisADenseDataset(
            root=args.visa_root,
            image_size=args.image_size,
            split=args.visa_train_split,
            split_csv=args.visa_split_csv,
        )
        train_ds = MaskDilateWrapper(base_train_ds, kernel_size=args.mask_dilate)
    elif args.train_dataset == "mvtec":
        # MVTec AD official train split contains only normal images.
        # For source-domain supervised cross-dataset training, we use the
        # dense MVTec loader over annotated samples. Since the target is VisA,
        # this does not use any target-domain samples.
        base_train_ds = MVTecDenseDataset(
            args.mvtec_root,
            image_size=args.image_size,
        )
        train_ds = MaskDilateWrapper(base_train_ds, kernel_size=args.mask_dilate)
    else:
        raise ValueError(f"Unknown train_dataset: {args.train_dataset}")

    labels_for_sampler = [s["label"] for s in train_ds.samples]
    num_normal = sum(1 for y in labels_for_sampler if y == 0)
    num_anom = sum(1 for y in labels_for_sampler if y == 1)

    sample_weights = [
        1.0 / num_normal if y == 0 else 1.0 / num_anom
        for y in labels_for_sampler
    ]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    print("normal samples:", num_normal)
    print("anomaly samples:", num_anom)
    print("using WeightedRandomSampler: approximately balanced normal/anomaly")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print("num train samples:", len(train_ds))
    print("classes:", train_ds.classes)

    print(f"===== {args.model_name} frozen extractor =====")
    if args.backbone_type == "clip_vit":
        clip_model, _, _ = open_clip.create_model_and_transforms(
            args.model_name,
            pretrained=args.pretrained,
        )
        clip_model = clip_model.to(device).eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        extractor = CLIPViTStageExtractor(
            clip_model,
            layers=(12,24),
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

    print("===== Text prototypes =====")
    tokenizer = open_clip.get_tokenizer(args.model_name)

    # Reuse the exact same frozen model inside extractor.
    prototypes = build_text_prototypes(
        model=extractor.model,
        tokenizer=tokenizer,
        class_names=train_ds.classes,
        device=device,
    )

    for cls in train_ds.classes[:5]:
        n, a = prototypes[cls]
        print(f"{cls}: normal {tuple(n.shape)}, abnormal {tuple(a.shape)}, cos={torch.dot(n, a).item():.4f}")

    learnable_prompt = None
    object_anchor_normal = None
    object_anchor_abnormal = None
    if args.learnable_prompt:
        print("===== Anchored Class-agnostic Learnable Prompt Adaptation =====")
        learnable_prompt = ClassAgnosticLearnablePrompt(
            text_tower=(extractor.model.text if hasattr(extractor.model, 'text') else extractor.model),
            tokenizer=tokenizer,
            class_names=list(prototypes.keys()),
            n_ctx=args.n_ctx,
            normal_word="normal",
            abnormal_word="defective",
            object_agnostic=args.object_agnostic_prompt,
            object_word=args.object_word,
        ).to(device)
        print(f"n_ctx: {args.n_ctx}")
        print(f"lambda_prompt_reg: {args.lambda_prompt_reg}")
        print(f"object_agnostic_prompt: {args.object_agnostic_prompt}")
        print(f"object_agnostic_anchor: {args.object_agnostic_anchor}")

        if args.object_agnostic_anchor:
            object_anchor_normal, object_anchor_abnormal = build_object_agnostic_anchor_prototypes(
                model=extractor.model,
                tokenizer=tokenizer,
                device=device,
                object_word=args.object_word,
            )
            print(
                "object anchor:",
                tuple(object_anchor_normal.shape),
                tuple(object_anchor_abnormal.shape),
                "cos=",
                float(torch.dot(object_anchor_normal, object_anchor_abnormal).detach().cpu()),
            )

    print("===== LDA / Dense Alignment Module =====")
    print("align_mode:", args.align_mode)

    head = DenseAlignHead(
        stage_dims=stage_dims,
        text_dim=text_dim,
        out_size=args.image_size,
        temperature_init=10.0,
        align_mode=args.align_mode,
        fixed_stage_fusion=args.fixed_stage_fusion,
    ).to(device)

    optim_params = list(head.parameters())

    if learnable_prompt is not None:

        optim_params += [learnable_prompt.normal_ctx, learnable_prompt.abnormal_ctx]


    optimizer = torch.optim.AdamW(optim_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=False)

    best_loss = float("inf")

    print("===== Train =====")
    for epoch in range(1, args.epochs + 1):
        head.train()

        meters = defaultdict(float)
        count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            class_names = batch["class_name"]

            if args.mask_dilate and args.mask_dilate > 1:
                train_masks = dilate_binary_mask(masks, args.mask_dilate)
            else:
                train_masks = masks

            text_normal, text_abnormal = collate_text_prototypes(
                class_names,
                prototypes,
                device,
            )
            prompt_anchor_loss = torch.tensor(0.0, device=device)

            if learnable_prompt is not None:
                batch_class_names = batch["class_name"]

                if args.object_agnostic_anchor:
                    fixed_text_normal = object_anchor_normal.unsqueeze(0).expand(len(batch_class_names), -1).detach()
                    fixed_text_abnormal = object_anchor_abnormal.unsqueeze(0).expand(len(batch_class_names), -1).detach()
                else:
                    fixed_text_normal = text_normal.detach()
                    fixed_text_abnormal = text_abnormal.detach()

                prompt_prototypes = learnable_prompt(batch_class_names)
                text_normal, text_abnormal = collate_text_prototypes(
                    batch_class_names,
                    prompt_prototypes,
                    device,
                )

                if args.lambda_prompt_anchor > 0:
                    prompt_anchor_loss = (
                        1.0 - F.cosine_similarity(text_normal, fixed_text_normal, dim=-1)
                    ).mean() + (
                        1.0 - F.cosine_similarity(text_abnormal, fixed_text_abnormal, dim=-1)
                    ).mean()

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                features = extractor(images)

            with torch.cuda.amp.autocast(enabled=False):
                out = head(features, text_normal, text_abnormal)
                logits = out["logits"]

                loss_focal = focal_loss_with_logits(logits, train_masks)
                loss_dice = dice_loss_with_logits(logits, train_masks, labels=labels)

                image_logits = topk_image_score(logits, ratio=args.topk_ratio)
                loss_img = F.binary_cross_entropy_with_logits(image_logits, labels.float())

                loss_cons = stage_consistency_loss(
                    out.get("stage_logits", None),
                    size=args.cons_size,
                )

                loss = (
                    args.lambda_focal * loss_focal
                    + args.lambda_dice * loss_dice
                    + args.lambda_img * loss_img
                    + args.lambda_cons * loss_cons
                )

            if learnable_prompt is not None and args.lambda_prompt_anchor > 0:
                loss = loss + args.lambda_prompt_anchor * prompt_anchor_loss

            if learnable_prompt is not None and args.lambda_prompt_reg > 0:
                loss = loss + args.lambda_prompt_reg * learnable_prompt.regularization_loss()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = images.shape[0]
            count += bs

            meters["loss"] += loss.item() * bs
            meters["focal"] += loss_focal.item() * bs
            meters["dice"] += loss_dice.item() * bs
            meters["img"] += loss_img.item() * bs
            meters["cons"] += loss_cons.item() * bs
            meters["logit_scale"] += out["logit_scale"].item() * bs

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                meters["mask_pos"] += masks.mean().item() * bs
                meters["prob_mean"] += probs.mean().item() * bs
                meters["prob_max"] += probs.amax(dim=(1, 2, 3)).mean().item() * bs
                meters["anom_ratio"] += labels.float().mean().item() * bs

            sw = out["stage_weights"].detach().cpu()
            for wi, s in enumerate(stages):
                meters[f"w{s}"] += sw[wi].item() * bs

            pbar.set_postfix({
                "loss": meters["loss"] / count,
                "focal": meters["focal"] / count,
                "dice": meters["dice"] / count,
                "img": meters["img"] / count,
                "cons": meters["cons"] / count,
                **{f"w{s}": meters[f"w{s}"] / count for s in stages},
                "mask": meters["mask_pos"] / count,
                "pmean": meters["prob_mean"] / count,
                "pmax": meters["prob_max"] / count,
                "ar": meters["anom_ratio"] / count,
            })

        epoch_loss = meters["loss"] / count

        log_line = (
            f"Epoch {epoch:03d} | "
            f"loss {meters['loss']/count:.6f} | "
            f"focal {meters['focal']/count:.6f} | "
            f"dice {meters['dice']/count:.6f} | "
            f"img {meters['img']/count:.6f} | "
            f"cons {meters['cons']/count:.6f} | "
            + " ".join([f"w{s} {meters[f'w{s}']/count:.4f} |" for s in stages]) + " "
            f"logit_scale {meters['logit_scale']/count:.4f} | "
            f"mask_pos {meters['mask_pos']/count:.6f} | "
            f"prob_mean {meters['prob_mean']/count:.6f} | "
            f"prob_max {meters['prob_max']/count:.6f} | "
            f"anom_ratio {meters['anom_ratio']/count:.4f}"
        )

        print(log_line)

        with open(output_dir / "train_log.txt", "a") as f:
            f.write(log_line + "\n")

        ckpt = {
            "epoch": epoch,
            "args": vars(args),
            "head": head.state_dict(),
            "learnable_prompt": None if learnable_prompt is None else {
                "normal_ctx": learnable_prompt.normal_ctx.detach().cpu(),
                "abnormal_ctx": learnable_prompt.abnormal_ctx.detach().cpu(),
            },
            "n_ctx": args.n_ctx,
            "object_agnostic_prompt": args.object_agnostic_prompt,
            "object_word": args.object_word,
            "object_agnostic_anchor": args.object_agnostic_anchor,
            "lambda_prompt_reg": args.lambda_prompt_reg,
            "optimizer": optimizer.state_dict(),
            "classes": train_ds.classes,
            "model_name": args.model_name,
                "backbone_type": args.backbone_type,
            "stage_dims": stage_dims,
            "text_dim": text_dim,
                    "align_mode": args.align_mode,
                    "fixed_stage_fusion": args.fixed_stage_fusion,
        }

        if epoch % args.save_every == 0:
            torch.save(ckpt, output_dir / f"epoch_{epoch:03d}.pth")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(ckpt, output_dir / "best.pth")
            print(f"saved best: {output_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
