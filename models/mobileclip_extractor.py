"""
Frozen MobileCLIP2 stage feature extractor.

This module wraps an open_clip MobileCLIP2 model and uses forward hooks to
capture hierarchical visual stage features such as stage2, stage3, and
stage4.
"""

from collections import OrderedDict
from typing import Dict, Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


class MobileCLIPStageExtractor(nn.Module):
    """
    MobileCLIP2-S3 frozen stage feature extractor.

    It uses forward hooks on visual.trunk.stages.{idx}.
    Expected MobileCLIP2-S3 feature shapes for 256x256 input:

        stage0: [B,  96, 64, 64]
        stage1: [B, 192, 32, 32]
        stage2: [B, 384, 16, 16]
        stage3: [B, 768,  8,  8]
        stage4: [B,1536,  4,  4]

    Main experiment uses stage2 and stage3.
    """

    def __init__(
        self,
        model_name: str = "MobileCLIP2-S3",
        pretrained: str = "dfndr2b",
        stages: Tuple[int, ...] = (2, 3),
        device: str = "cuda",
        freeze: bool = True,
    ):
        super().__init__()

        self.model_name = model_name
        self.pretrained = pretrained
        self.stages = tuple(stages)
        self.device_name = device

        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )

        self.model = model
        self.model.eval()

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

        self._features: Dict[int, torch.Tensor] = OrderedDict()
        self._hooks: List = []
        self._register_stage_hooks()

    def _find_stage_module(self, stage_idx: int):
        """
        Find module named like:
            visual.trunk.stages.2
            trunk.stages.2
            stages.2

        This is intentionally flexible because open_clip internals
        can slightly differ across versions.
        """
        candidates = []

        for name, module in self.model.named_modules():
            parts = name.split(".")
            if len(parts) >= 2 and parts[-2] == "stages" and parts[-1] == str(stage_idx):
                candidates.append((name, module))

        if len(candidates) == 0:
            raise RuntimeError(
                f"Could not find stage {stage_idx} module. "
                f"Run the debug block in this file to inspect named_modules()."
            )

        # Prefer the shortest exact stage module path, not inner blocks.
        candidates = sorted(candidates, key=lambda x: len(x[0]))
        return candidates[0]

    def _register_stage_hooks(self):
        for s in self.stages:
            name, module = self._find_stage_module(s)

            def make_hook(stage_idx):
                def hook(_module, _input, output):
                    if isinstance(output, (tuple, list)):
                        output = output[0]
                    self._features[stage_idx] = output
                return hook

            h = module.register_forward_hook(make_hook(s))
            self._hooks.append(h)
            print(f"[MobileCLIPStageExtractor] hooked stage{s}: {name}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Returns:
            dict:
                stage_idx -> feature tensor [B, C, H, W]
        """
        self._features = OrderedDict()

        # Running encode_image is usually enough to trigger visual trunk stages.
        _ = self.model.encode_image(x)

        out = OrderedDict()
        for s in self.stages:
            if s not in self._features:
                raise RuntimeError(
                    f"Stage {s} was not captured. "
                    f"Available captured stages: {list(self._features.keys())}"
                )

            feat = self._features[s]

            # Some modules may return NHWC. Convert to NCHW if needed.
            if feat.ndim != 4:
                raise RuntimeError(f"Expected 4D feature from stage{s}, got shape {tuple(feat.shape)}")

            # Heuristic: if channel dim is last and second dim looks spatial, permute.
            if feat.shape[1] <= 64 and feat.shape[-1] > 64:
                feat = feat.permute(0, 3, 1, 2).contiguous()

            out[s] = feat

        return out

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


def print_stage_like_modules(model_name="MobileCLIP2-S3", pretrained="dfndr2b"):
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)

    print("===== stage-like modules =====")
    for name, module in model.named_modules():
        if "stage" in name.lower() or "stages" in name.lower():
            print(name, "->", module.__class__.__name__)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("device:", device)

    try:
        extractor = MobileCLIPStageExtractor(
            model_name="MobileCLIP2-S3",
            pretrained="dfndr2b",
            stages=(0, 1, 2, 3, 4),
            device=device,
        ).to(device)

        x = torch.randn(2, 3, 256, 256, device=device)
        feats = extractor(x)

        print("===== captured features =====")
        for k, v in feats.items():
            print(f"stage{k}:", tuple(v.shape))

    except Exception as e:
        print("ERROR:", repr(e))
        print()
        print_stage_like_modules()
