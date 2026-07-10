"""
Dense alignment modules for MobileCLIP-AD.

The final paper setting is LDA (Lightweight Dense Alignment), implemented as
MobileCLIP2 hierarchical stage features for dense text-guided anomaly
segmentation. LDA extends a projection-only alignment head with lightweight
local spatial refinement.

Projection-only alignment is kept as an ablation baseline.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StageProjectionHead(nn.Module):
    """
    Stage-wise dense alignment block.

    The final LDA setting is align_mode="local":
        projection -> local spatial refinement.

    Modes:
        projection: projection-only alignment
        local:      projection + depthwise local refinement

    LDA is the dense alignment module used in the final MobileCLIP-AD.
    """

    def __init__(
        self,
        in_dim: int,
        text_dim: int = 768,
        num_groups: int = 32,
        align_mode: str = "projection",
        fixed_stage_fusion: bool = False
    ):
        super().__init__()

        assert align_mode in ["projection", "local"], (
            f"Unknown align_mode: {align_mode}"
        )

        self.align_mode = align_mode
        self.fixed_stage_fusion = bool(fixed_stage_fusion)
        self.use_local_refine = align_mode == "local"
        self.text_dim = text_dim

        # num_groups must divide text_dim.
        if text_dim % num_groups != 0:
            num_groups = 1

        # Original projection-only path.
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, text_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=text_dim),
            nn.GELU(),
            nn.Conv2d(text_dim, text_dim, kernel_size=1, bias=True),
        )

        # Lightweight local spatial refinement.
        # Depthwise 3x3 keeps the parameter/FLOP overhead small.
        if self.use_local_refine:
            self.local_refine = nn.Sequential(
                nn.Conv2d(
                    text_dim,
                    text_dim,
                    kernel_size=3,
                    padding=1,
                    groups=text_dim,
                    bias=False,
                ),
                nn.GELU(),
                nn.Conv2d(text_dim, text_dim, kernel_size=1, bias=True),
            )
            # Identity initialization: z + local_refine(z) starts as z.
            nn.init.zeros_(self.local_refine[-1].weight)
            nn.init.zeros_(self.local_refine[-1].bias)
        else:
            self.local_refine = None

    def _expand_text(self, text: torch.Tensor, batch_size: int) -> torch.Tensor:
        if text.ndim == 1:
            text = text.unsqueeze(0)
        if text.shape[0] == 1 and batch_size > 1:
            text = text.expand(batch_size, -1)
        return text

    def forward(
        self,
        x: torch.Tensor,
        text_normal: torch.Tensor = None,
        text_abnormal: torch.Tensor = None,
    ) -> torch.Tensor:
        B = x.shape[0]

        z = self.proj(x)

        if self.use_local_refine:
            z = z + self.local_refine(z)

        z = F.normalize(z, dim=1)
        return z


class DenseAlignHead(nn.Module):
    """
    Convert dense backbone stage features into LDA-aligned anomaly maps.

    Input:
        features:
            {
                2: [B, 384, 16, 16],
                3: [B, 768, 8, 8],
            }

        text_normal:   [B, 768] or [768]
        text_abnormal: [B, 768] or [768]

    Output:
        {
            "logits": [B, 1, out_size, out_size],
            "stage_logits": {2: ..., 3: ...},
            "stage_weights": [num_stages]
        }
    """

    def __init__(
        self,
        stage_dims: Dict[int, int],
        text_dim: int = 768,
        out_size: int = 256,
        temperature_init: float = 10.0,
        align_mode: str = "projection",
        fixed_stage_fusion: bool = False
    ):
        super().__init__()

        self.stage_ids = tuple(sorted(stage_dims.keys()))
        self.text_dim = text_dim
        self.out_size = out_size

        self.align_mode = align_mode
        self.fixed_stage_fusion = bool(fixed_stage_fusion)

        self.proj = nn.ModuleDict({
            str(s): StageProjectionHead(
                stage_dims[s],
                text_dim=text_dim,
                align_mode=align_mode
            )
            for s in self.stage_ids
        })

        # Stage fusion. Final LDA uses learnable fusion; projection-only ablation can use fixed equal fusion.
        self.stage_logits = nn.Parameter(torch.zeros(len(self.stage_ids)))

        # Learnable scaling for anomaly logits.
        # Start moderately sharp but not CLIP logit_scale-level huge.
        self.logit_scale = nn.Parameter(torch.tensor(float(temperature_init)).log())

        # Optional global bias helps calibration.
        self.bias = nn.Parameter(torch.zeros(1))

    def _prepare_text(self, text: torch.Tensor, batch_size: int) -> torch.Tensor:
        if text.ndim == 1:
            text = text.unsqueeze(0)
        if text.shape[0] == 1 and batch_size > 1:
            text = text.expand(batch_size, -1)
        text = F.normalize(text, dim=-1)
        return text

    def _cosine_map(self, dense_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """
        dense_feat: [B, C, H, W], already normalized
        text_feat:  [B, C], normalized

        return:
            [B, 1, H, W]
        """
        return torch.einsum("bchw,bc->bhw", dense_feat, text_feat).unsqueeze(1)

    def forward(
        self,
        features: Dict[int, torch.Tensor],
        text_normal: torch.Tensor,
        text_abnormal: torch.Tensor,
    ):
        first_stage = self.stage_ids[0]
        B = features[first_stage].shape[0]

        text_normal = self._prepare_text(text_normal, B)
        text_abnormal = self._prepare_text(text_abnormal, B)

        if self.fixed_stage_fusion:
            weights = torch.ones(
                len(self.stage_ids),
                device=self.stage_logits.device,
                dtype=self.stage_logits.dtype,
            ) / len(self.stage_ids)
        else:
            weights = F.softmax(self.stage_logits, dim=0)
        scale = self.logit_scale.exp().clamp(1.0, 100.0)

        fused = 0.0
        stage_out = {}

        for i, s in enumerate(self.stage_ids):
            feat = features[s]
            dense = self.proj[str(s)](feat, text_normal, text_abnormal)

            sim_n = self._cosine_map(dense, text_normal)
            sim_a = self._cosine_map(dense, text_abnormal)

            # anomaly evidence
            logit = (sim_a - sim_n) * scale + self.bias

            logit_up = F.interpolate(
                logit,
                size=(self.out_size, self.out_size),
                mode="bilinear",
                align_corners=False,
            )

            stage_out[s] = logit_up
            fused = fused + weights[i] * logit_up

        return {
            "logits": fused,
            "stage_logits": stage_out,
            "stage_weights": weights.detach(),
            "logit_scale": scale.detach(),
        }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    head = DenseAlignHead(
        stage_dims={2: 384, 3: 768},
        text_dim=768,
        out_size=256,
    ).to(device)

    feats = {
        2: torch.randn(2, 384, 16, 16, device=device),
        3: torch.randn(2, 768, 8, 8, device=device),
    }

    tn = torch.randn(2, 768, device=device)
    ta = torch.randn(2, 768, device=device)

    out = head(feats, tn, ta)
    print("logits:", tuple(out["logits"].shape))
    for k, v in out["stage_logits"].items():
        print("stage", k, tuple(v.shape))
    print("stage weights:", out["stage_weights"])
    print("logit scale:", out["logit_scale"])
