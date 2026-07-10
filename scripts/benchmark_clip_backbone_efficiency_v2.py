import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import open_clip


def count_params(module: nn.Module):
    return sum(p.numel() for p in module.parameters())


def try_fvcore_macs(module, x):
    try:
        from fvcore.nn import FlopCountAnalysis
        module.eval()
        with torch.no_grad():
            macs = FlopCountAnalysis(module, x).total()
        return macs
    except Exception as e:
        print(f"[WARN] fvcore MACs failed: {type(e).__name__}: {e}")
        return None


class VisualOnlyWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.visual = model.visual

    def forward(self, x):
        return self.visual(x)


def estimate_text_params(model, total_params, visual_params):
    """
    Robust fallback:
    For standard open_clip CLIP, text-side module names are visible.
    For MobileCLIP2, they may not be exposed under the same names, so use total - visual.
    """
    text_names = [
        "token_embedding",
        "transformer",
        "positional_embedding",
        "ln_final",
        "text_projection",
    ]

    params = []
    seen = set()

    for name in text_names:
        if not hasattr(model, name):
            continue
        obj = getattr(model, name)
        if isinstance(obj, torch.nn.Parameter):
            if id(obj) not in seen:
                params.append(obj)
                seen.add(id(obj))
        elif isinstance(obj, torch.nn.Module):
            for p in obj.parameters():
                if id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))

    text_params = sum(p.numel() for p in params)

    # MobileCLIP2 fallback
    if text_params == 0 and visual_params is not None:
        text_params = total_params - visual_params

    return text_params


def benchmark_one(display_name, model_name, pretrained, image_size, device):
    print("=" * 80)
    print(f"[MODEL] {display_name} | open_clip={model_name} | pretrained={pretrained} | image_size={image_size}")
    print("=" * 80)

    model = open_clip.create_model(
        model_name,
        pretrained=pretrained if pretrained.lower() != "none" else None,
    )

    model = model.to(device).eval()

    total_params = count_params(model)
    visual_params = count_params(model.visual) if hasattr(model, "visual") else None
    text_params = estimate_text_params(model, total_params, visual_params)

    x = torch.randn(1, 3, image_size, image_size).to(device)
    visual = VisualOnlyWrapper(model).to(device).eval()

    macs = try_fvcore_macs(visual, x)
    flops = macs * 2 if macs is not None else None

    n_warmup = 10
    n_iter = 50

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = visual(x)

        if device == "cuda":
            torch.cuda.synchronize()

        import time
        t0 = time.time()

        for _ in range(n_iter):
            _ = visual(x)

        if device == "cuda":
            torch.cuda.synchronize()

        t1 = time.time()

    latency_ms = (t1 - t0) / n_iter * 1000.0

    row = {
        "display_name": display_name,
        "open_clip_model": model_name,
        "pretrained": pretrained,
        "image_size": image_size,
        "total_params_M": total_params / 1e6,
        "visual_params_M": visual_params / 1e6 if visual_params is not None else None,
        "text_params_M": text_params / 1e6 if text_params is not None else None,
        "visual_macs_G": macs / 1e9 if macs is not None else None,
        "visual_flops_G_muladd2": flops / 1e9 if flops is not None else None,
        "latency_ms_bs1": latency_ms,
    }

    print(pd.DataFrame([row]).to_string(index=False))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_csv", type=str, default="paper_results/source_csv/clip_backbone_efficiency_v2.csv")
    args = parser.parse_args()

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    specs = [
        ("CLIP ViT-B/32 @224", "ViT-B-32", "openai", 224),
        ("CLIP ViT-B/16 @224", "ViT-B-16", "openai", 224),
        ("CLIP ViT-L/14 @224", "ViT-L-14", "openai", 224),
        ("CLIP ViT-L/14 @336", "ViT-L-14-336", "openai", 336),
        ("MobileCLIP2-S3 @256", "MobileCLIP2-S3", "dfndr2b", 256),
        ("MobileCLIP2-S4 @256", "MobileCLIP2-S4", "dfndr2b", 256),
    ]

    rows = []

    for display_name, model_name, pretrained, image_size in specs:
        try:
            rows.append(benchmark_one(display_name, model_name, pretrained, image_size, args.device))
        except Exception as e:
            print(f"[ERROR] failed: {display_name}: {type(e).__name__}: {e}")
            rows.append({
                "display_name": display_name,
                "open_clip_model": model_name,
                "pretrained": pretrained,
                "image_size": image_size,
                "error": f"{type(e).__name__}: {e}",
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)

    print("")
    print("=" * 80)
    print("[SUMMARY]")
    print("=" * 80)

    show_cols = [
        "display_name",
        "image_size",
        "total_params_M",
        "visual_params_M",
        "text_params_M",
        "visual_macs_G",
        "visual_flops_G_muladd2",
        "latency_ms_bs1",
        "error",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    print(df[show_cols].to_string(index=False))

    print("")
    print("saved:", args.output_csv)


if __name__ == "__main__":
    main()
