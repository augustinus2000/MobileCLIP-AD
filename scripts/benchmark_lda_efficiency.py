import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


import pandas as pd
import torch

from models.densealign import DenseAlignHead


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def count_trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def infer_head_params(stage_dims, text_dim, align_mode="local"):
    head = DenseAlignHead(
        stage_dims=stage_dims,
        text_dim=text_dim,
        align_mode=align_mode,
    )
    return count_params(head), count_trainable(head)


def prompt_params(text_dim, n_ctx=4, num_prompts=2):
    # normal prompt + abnormal prompt
    return n_ctx * text_dim * num_prompts


def main():
    # Backbone params/FLOPs are kept from the already measured backbone benchmark.
    # This script recomputes LDA-head and prompt trainable parameters under align_mode="local".
    rows = [
        {
            "Backbone": "MobileCLIP2-S3@256",
            "Total Params (M)": 248.89,
            "Image Params (M)": 125.24,
            "Text Params (M)": 123.65,
            "stage_dims": {2: 384, 3: 768},
            "text_dim": 768,
            "Image FLOPs (G)": 29.68,
            "Latency (ms)": 14.60,
        },
        {
            "Backbone": "CLIP ViT-L/14@224",
            "Total Params (M)": 427.62,
            "Image Params (M)": 303.97,
            "Text Params (M)": 123.65,
            "stage_dims": {2: 1024, 3: 1024},
            "text_dim": 768,
            "Image FLOPs (G)": 116.73,
            "Latency (ms)": 15.20,
        },
        {
            "Backbone": "CLIP ViT-L/14@336",
            "Total Params (M)": 427.94,
            "Image Params (M)": 304.29,
            "Text Params (M)": 123.65,
            "stage_dims": {2: 1024, 3: 1024},
            "text_dim": 768,
            "Image FLOPs (G)": 262.07,
            "Latency (ms)": 29.46,
        },
    ]

    out_rows = []
    for r in rows:
        head_params, head_trainable = infer_head_params(
            r["stage_dims"],
            r["text_dim"],
            align_mode="local",
        )
        p_params = prompt_params(r["text_dim"], n_ctx=4, num_prompts=2)

        trainable = head_trainable + p_params

        out_rows.append({
            "Backbone": r["Backbone"],
            "Total Params (M)": r["Total Params (M)"],
            "Image Params (M)": r["Image Params (M)"],
            "Text Params (M)": r["Text Params (M)"],
            "LDA Head (M)": head_params / 1e6,
            "Prompt (M)": p_params / 1e6,
            "Trainable (M)": trainable / 1e6,
            "Image FLOPs (G)": r["Image FLOPs (G)"],
            "Latency (ms)": r["Latency (ms)"],
        })

    df = pd.DataFrame(out_rows)

    out_dir = Path("paper_results/source_csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    numeric_path = out_dir / "lda_backbone_efficiency_numeric.csv"
    paper_path = out_dir / "lda_backbone_efficiency_paper.csv"

    df.to_csv(numeric_path, index=False)

    paper = df.copy()
    for c in paper.columns:
        if c != "Backbone":
            paper[c] = paper[c].map(lambda x: f"{x:.2f}")
    paper.to_csv(paper_path, index=False)

    print("Saved:", numeric_path)
    print("Saved:", paper_path)
    print()
    print(paper.to_string(index=False))


if __name__ == "__main__":
    main()
