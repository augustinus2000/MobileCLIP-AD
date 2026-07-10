import torch
import torch.nn as nn


class CLIPViTStageExtractor(nn.Module):
    """
    Dense feature extractor for OpenCLIP ViT.

    Returns same interface as MobileCLIPStageExtractor:
        {
            2: mid-level patch tokens,
            3: high-level patch tokens
        }

    Note:
    These are not equivalent to MobileCLIP stages.
    They are backbone-specific dense features.
    """

    def __init__(self, model, layers=(12, 24)):
        super().__init__()

        self.visual = model.visual
        self.layers = layers
        self.outputs = {}

        blocks = self.visual.transformer.resblocks

        for layer in layers:
            blocks[layer - 1].register_forward_hook(
                self._hook(layer)
            )

        print(f"[CLIPViTStageExtractor] hooked transformer layers {layers}")

    def _hook(self, layer):
        def fn(module, inp, out):

            # OpenCLIP ViT hook output can be either:
            # [batch, sequence, channel] or [sequence, batch, channel]
            if out.shape[0] == 257:
                x = out.permute(1, 0, 2)
            else:
                x = out

            # remove CLS token
            x = x[:, 1:]

            B, N, C = x.shape
            H = W = int(N ** 0.5)

            x = (
                x.reshape(B, H, W, C)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

            self.outputs[layer] = x

        return fn

    @torch.no_grad()
    def forward(self, x):

        self.outputs = {}

        _ = self.visual(x)

        return {
            2: self.outputs[self.layers[0]],
            3: self.outputs[self.layers[1]],
        }
