import os
import sys
import time
import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import open_clip

from models.densealign import DenseAlignHead
from models.learnable_prompt import ClassAgnosticLearnablePrompt


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def count_trainable_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def try_get_visual_text(model):
    visual = getattr(model, "visual", None)
    text = getattr(model, "text", None)
    return visual, text


def make_stage_dims(model_name):
    stage_dims = {
        "MobileCLIP2-S0": {2: 256, 3: 512},
        "MobileCLIP2-S2": {2: 320, 3: 640},
        "MobileCLIP2-S3": {2: 384, 3: 768},
        "MobileCLIP2-S4": {2: 512, 3: 1024},
    }
    text_dims = {
        "MobileCLIP2-S0": 512,
        "MobileCLIP2-S2": 512,
        "MobileCLIP2-S3": 768,
        "MobileCLIP2-S4": 768,
    }
    return stage_dims[model_name], text_dims[model_name]


def benchmark_latency(model, image_size=256, batch_size=1, device="cuda", warmup=20, repeat=100):
    model.eval()
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model.encode_image(x)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(repeat):
            _ = model.encode_image(x)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    elapsed = time.time() - start
    latency_ms = elapsed / repeat * 1000.0
    fps = batch_size * repeat / elapsed
    return latency_ms, fps


def benchmark_flops_with_thop(model, image_size=256, device="cuda"):
    try:
        from thop import profile
    except Exception:
        return None, None, "thop_not_installed"

    class ImageEncoderWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model.encode_image(x)

    wrapper = ImageEncoderWrapper(model).to(device).eval()
    x = torch.randn(1, 3, image_size, image_size, device=device)

    try:
        macs, params = profile(wrapper, inputs=(x,), verbose=False)
        flops = macs * 2
        return float(macs), float(flops), "ok"
    except Exception as e:
        return None, None, f"failed: {type(e).__name__}: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", type=str, default="dfndr2b")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="paper_results/source_csv/mobileclip2_variant_efficiency.csv")
    parser.add_argument("--latency_repeat", type=int, default=100)
    parser.add_argument("--latency_warmup", type=int, default=20)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"

    models = [
        "MobileCLIP2-S0",
        "MobileCLIP2-S2",
        "MobileCLIP2-S3",
        "MobileCLIP2-S4",
    ]

    rows = []

    for model_name in models:
        print(f"\n===== {model_name} =====")

        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=args.pretrained,
        )
        model = model.to(device).eval()

        for p in model.parameters():
            p.requires_grad = False

        tokenizer = open_clip.get_tokenizer(model_name)

        stage_dims, text_dim = make_stage_dims(model_name)

        head = DenseAlignHead(
            stage_dims=stage_dims,
            text_dim=text_dim,
            out_size=args.image_size,
            temperature_init=10.0,
            align_mode="sgda",
        ).to(device)

        prompt = ClassAgnosticLearnablePrompt(
            text_tower=model.text,
            tokenizer=tokenizer,
            class_names=["object"],
            n_ctx=4,
            normal_word="normal",
            abnormal_word="defective",
            object_agnostic=True,
            object_word="object",
        ).to(device)

        visual, text = try_get_visual_text(model)

        total_params = count_params(model)
        visual_params = count_params(visual) if visual is not None else None
        text_params = count_params(text) if text is not None else None

        head_params = count_params(head)
        head_trainable = count_trainable_params(head)

        prompt_params = count_params(prompt)
        prompt_trainable = prompt.normal_ctx.numel() + prompt.abnormal_ctx.numel()

        trainable_ad_params = head_trainable + prompt_trainable

        macs, flops, flops_status = benchmark_flops_with_thop(
            model=model,
            image_size=args.image_size,
            device=device,
        )

        latency_ms, fps = benchmark_latency(
            model=model,
            image_size=args.image_size,
            batch_size=args.batch_size,
            device=device,
            warmup=args.latency_warmup,
            repeat=args.latency_repeat,
        )

        row = {
            "model": model_name,
            "image_size": args.image_size,
            "backbone_total_params": total_params,
            "visual_params": visual_params,
            "text_params": text_params,
            "sgda_head_params": head_params,
            "sgda_head_trainable_params": head_trainable,
            "prompt_trainable_params": prompt_trainable,
            "total_trainable_ad_params": trainable_ad_params,
            "macs": macs,
            "flops": flops,
            "flops_status": flops_status,
            "latency_ms_per_image_encoder": latency_ms,
            "fps_image_encoder": fps,
        }

        rows.append(row)

        print(f"backbone_total_params: {total_params/1e6:.2f} M")
        if visual_params is not None:
            print(f"visual_params:         {visual_params/1e6:.2f} M")
        if text_params is not None:
            print(f"text_params:           {text_params/1e6:.2f} M")
        print(f"sgda_head_params:      {head_params/1e6:.3f} M")
        print(f"prompt_trainable:      {prompt_trainable/1e3:.1f} K")
        print(f"AD trainable params:   {trainable_ad_params/1e6:.3f} M")
        if macs is not None:
            print(f"MACs:                  {macs/1e9:.3f} G")
            print(f"FLOPs:                 {flops/1e9:.3f} G")
        else:
            print(f"FLOPs status:          {flops_status}")
        print(f"latency:               {latency_ms:.2f} ms/img")
        print(f"FPS:                   {fps:.2f}")

        del model, head, prompt
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    pretty = df.copy()

    for col in [
        "backbone_total_params",
        "visual_params",
        "text_params",
        "sgda_head_params",
        "sgda_head_trainable_params",
        "total_trainable_ad_params",
    ]:
        if col in pretty.columns:
            pretty[col] = pretty[col].apply(lambda x: None if pd.isna(x) else x / 1e6)

    pretty["prompt_trainable_params"] = pretty["prompt_trainable_params"] / 1e3

    if "macs" in pretty.columns:
        pretty["macs"] = pretty["macs"].apply(lambda x: None if pd.isna(x) else x / 1e9)
    if "flops" in pretty.columns:
        pretty["flops"] = pretty["flops"].apply(lambda x: None if pd.isna(x) else x / 1e9)

    print("\n===== Pretty Summary =====")
    cols = [
        "model",
        "backbone_total_params",
        "visual_params",
        "text_params",
        "sgda_head_params",
        "prompt_trainable_params",
        "total_trainable_ad_params",
        "macs",
        "flops",
        "latency_ms_per_image_encoder",
        "fps_image_encoder",
        "flops_status",
    ]
    print(pretty[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
