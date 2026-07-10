"""
Class-agnostic learnable prompt module for MobileCLIP2 text tower.

This module implements a CoOp-style continuous prompt for normal and abnormal
text prototypes. The learnable context tokens are shared across all classes,
while the suffix tokens such as "normal bottle" or "defective bottle" remain
class-aware and fixed.

Prompt format:
    SOS + normal_ctx   + "normal {class}"   + EOT + PAD
    SOS + abnormal_ctx + "defective {class}" + EOT + PAD

Only normal_ctx and abnormal_ctx are learnable.
The text tower itself should remain frozen.
"""

from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassAgnosticLearnablePrompt(nn.Module):
    """
    Class-agnostic learnable context tokens for normal/abnormal prototypes.

    Parameters
    ----------
    text_tower:
        model.text from open_clip MobileCLIP2.
    tokenizer:
        open_clip tokenizer.
    class_names:
        Iterable of class names.
    n_ctx:
        Number of learnable context tokens.
    normal_word:
        Fixed suffix word for normal prototypes.
    abnormal_word:
        Fixed suffix word for abnormal prototypes.
    ctx_init_std:
        Standard deviation for random context initialization.
    """

    def __init__(
        self,
        text_tower,
        tokenizer,
        class_names: Iterable[str],
        n_ctx: int = 4,
        normal_word: str = "normal",
        abnormal_word: str = "defective",
        ctx_init_std: float = 0.02,
        object_agnostic: bool = False,
        object_word: str = "object",
    ):
        super().__init__()

        self.text_tower = text_tower
        self.tokenizer = tokenizer
        self.class_names = list(class_names)
        self.n_ctx = int(n_ctx)
        self.normal_word = normal_word
        self.abnormal_word = abnormal_word
        self.object_agnostic = bool(object_agnostic)
        self.object_word = object_word

        width = int(text_tower.token_embedding.weight.shape[1])
        self.width = width
        self.context_length = int(text_tower.positional_embedding.shape[0])

        self.normal_ctx = nn.Parameter(torch.randn(self.n_ctx, width) * ctx_init_std)
        self.abnormal_ctx = nn.Parameter(torch.randn(self.n_ctx, width) * ctx_init_std)

        for p in self.text_tower.parameters():
            p.requires_grad = False

    @staticmethod
    def _clean_class_name(class_name: str) -> str:
        return class_name.replace("_", " ")

    def _make_suffix_tokens(self, class_names: List[str], mode: str, device: torch.device):
        """
        Build suffix token ids excluding SOS but including EOT.

        Example:
            tokenizer("normal bottle") -> [SOS, normal, bottle, EOT, PAD...]
            suffix = [normal, bottle, EOT]
        """
        assert mode in {"normal", "abnormal"}

        word = self.normal_word if mode == "normal" else self.abnormal_word
        if self.object_agnostic:
            texts = [f"{word} {self.object_word}" for _ in class_names]
        else:
            texts = [f"{word} {self._clean_class_name(c)}" for c in class_names]
        tokens = self.tokenizer(texts).to(device)

        suffixes = []
        lengths = []
        eot_token_id = int(tokens.max().item())

        for row in tokens:
            eot_pos = int((row == eot_token_id).nonzero(as_tuple=False)[0].item())
            suffix = row[1 : eot_pos + 1]  # exclude SOS, include EOT
            suffixes.append(suffix)
            lengths.append(int(suffix.numel()))

        return suffixes, lengths

    def _encode_with_ctx(self, class_names: List[str], ctx: torch.Tensor, mode: str):
        device = ctx.device
        suffixes, lengths = self._make_suffix_tokens(class_names, mode=mode, device=device)

        bsz = len(class_names)
        pad_id = 0

        # Start from PAD token embeddings, same sequence length as CLIP text input.
        pad_tokens = torch.full(
            (bsz, self.context_length),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        x = self.text_tower.token_embedding(pad_tokens)

        # SOS token id is the first token produced by tokenizer.
        sos_tokens = self.tokenizer([""])[0:1, 0].to(device)
        sos_embed = self.text_tower.token_embedding(sos_tokens)[0]

        eot_positions = []
        for i, suffix in enumerate(suffixes):
            suffix_len = lengths[i]
            eot_pos = 1 + self.n_ctx + suffix_len - 1

            if eot_pos >= self.context_length:
                raise RuntimeError(
                    f"Prompt too long for context_length={self.context_length}: "
                    f"class={class_names[i]}, eot_pos={eot_pos}"
                )

            x[i, 0] = sos_embed
            x[i, 1 : 1 + self.n_ctx] = ctx
            x[i, 1 + self.n_ctx : 1 + self.n_ctx + suffix_len] = (
                self.text_tower.token_embedding(suffix)
            )
            eot_positions.append(eot_pos)

        pos = self.text_tower.positional_embedding[: self.context_length].to(x.dtype)
        x = x + pos

        try:
            x = self.text_tower.transformer(
                x,
                attn_mask=getattr(self.text_tower, "attn_mask", None),
            )
        except TypeError:
            x = self.text_tower.transformer(x)

        x = self.text_tower.ln_final(x)

        eot_positions = torch.tensor(eot_positions, dtype=torch.long, device=device)
        x = x[torch.arange(bsz, device=device), eot_positions]

        if getattr(self.text_tower, "text_projection", None) is not None:
            x = x @ self.text_tower.text_projection

        x = F.normalize(x, dim=-1)
        return x

    def forward(self, class_names: Iterable[str] = None) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Return a dictionary:
            class_name -> (normal_proto, abnormal_proto)

        Each prototype has shape [D].
        """
        if class_names is None:
            class_names = self.class_names
        class_names = list(class_names)

        normal = self._encode_with_ctx(class_names, self.normal_ctx, mode="normal")
        abnormal = self._encode_with_ctx(class_names, self.abnormal_ctx, mode="abnormal")

        return {
            cls: (normal[i], abnormal[i])
            for i, cls in enumerate(class_names)
        }

    def regularization_loss(self):
        return self.normal_ctx.pow(2).mean() + self.abnormal_ctx.pow(2).mean()
