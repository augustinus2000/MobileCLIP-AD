"""
Industrial normal/abnormal prompt ensemble utilities.

This module builds class-specific normal and abnormal text prompts and
converts them into normalized text prototypes using the frozen
MobileCLIP2 text encoder.
"""

from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
import open_clip


NORMAL_TEMPLATES = [
    "a photo of a normal {}",
    "a photo of a flawless {}",
    "a photo of an undamaged {}",
    "a photo of a defect-free {}",
    "a photo of a clean {}",
]

ABNORMAL_TEMPLATES = [
    "a photo of a defective {}",
    "a photo of a damaged {}",
    "a photo of a flawed {}",
    "a photo of an abnormal {}",
    "a photo of a {} with a defect",
]


def normalize_class_name(class_name: str) -> str:
    return class_name.replace("_", " ")


@torch.no_grad()
def build_text_prototypes(
    model,
    tokenizer,
    class_names: List[str],
    device: str = "cuda",
    normal_templates: List[str] = NORMAL_TEMPLATES,
    abnormal_templates: List[str] = ABNORMAL_TEMPLATES,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Returns:
        prototypes[class_name] = (normal_proto, abnormal_proto)
        each proto: [text_dim]
    """
    model.eval()
    prototypes = {}

    for cls in class_names:
        name = normalize_class_name(cls)

        normal_prompts = [t.format(name) for t in normal_templates]
        abnormal_prompts = [t.format(name) for t in abnormal_templates]

        normal_tokens = tokenizer(normal_prompts).to(device)
        abnormal_tokens = tokenizer(abnormal_prompts).to(device)

        normal_feats = model.encode_text(normal_tokens)
        abnormal_feats = model.encode_text(abnormal_tokens)

        normal_feats = F.normalize(normal_feats, dim=-1)
        abnormal_feats = F.normalize(abnormal_feats, dim=-1)

        normal_proto = F.normalize(normal_feats.mean(dim=0), dim=0)
        abnormal_proto = F.normalize(abnormal_feats.mean(dim=0), dim=0)

        prototypes[cls] = (normal_proto.detach(), abnormal_proto.detach())

    return prototypes


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, _ = open_clip.create_model_and_transforms(
        "MobileCLIP2-S3",
        pretrained="dfndr2b",
    )
    model = model.to(device).eval()

    tokenizer = open_clip.get_tokenizer("MobileCLIP2-S3")

    protos = build_text_prototypes(
        model=model,
        tokenizer=tokenizer,
        class_names=["bottle", "metal_nut", "candle", "chewinggum"],
        device=device,
    )

    for k, (n, a) in protos.items():
        print(k, n.shape, a.shape, "cos(n,a)=", torch.dot(n, a).item())
