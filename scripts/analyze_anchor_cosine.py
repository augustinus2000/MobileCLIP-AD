import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import open_clip

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.learnable_prompt import ClassAgnosticLearnablePrompt


def get_text_tower(model):
    # MobileCLIP2 in this repo usually exposes model.text.
    # Some open_clip CLIP models expose text components directly, but here we focus on MobileCLIP2.
    if hasattr(model, "text"):
        return model.text
    raise AttributeError("model.text not found. Check MobileCLIP2 text tower attribute.")


@torch.no_grad()
def encode_fixed_text(model, tokenizer, texts, device):
    tokens = tokenizer(texts).to(device)
    try:
        x = model.encode_text(tokens, normalize=True)
    except TypeError:
        x = model.encode_text(tokens)
        x = F.normalize(x, dim=-1)
    return F.normalize(x.float(), dim=-1)


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "MobileCLIP2-S3"
    pretrained = "dfndr2b"

    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    text_tower = get_text_tower(model)

    fixed = encode_fixed_text(
        model,
        tokenizer,
        ["normal object", "defective object"],
        device,
    )
    anchor_n = fixed[0]
    anchor_a = fixed[1]

    rows = []

    for lam in ["0.0", "0.25", "0.5", "0.75", "1.0"]:
        if lam == "0.5":
            ckpt_path = Path("outputs/lda_s3_objprompt_objanchor_seed0_visa/best.pth")
        else:
            tag_map = {
                "0.0": "0p00",
                "0.25": "0p25",
                "0.75": "0p75",
                "1.0": "1p00",
            }
            tag = tag_map[lam]
            ckpt_path = Path(f"outputs/lda_anchor_s3_anchor{tag}_seed0/best.pth")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        prompt = ClassAgnosticLearnablePrompt(
            text_tower=text_tower,
            tokenizer=tokenizer,
            class_names=["object"],
            n_ctx=ckpt["n_ctx"],
            normal_word="normal",
            abnormal_word="defective",
            object_agnostic=True,
            object_word="object",
        ).to(device)

        prompt.normal_ctx.data.copy_(ckpt["learnable_prompt"]["normal_ctx"].to(device))
        prompt.abnormal_ctx.data.copy_(ckpt["learnable_prompt"]["abnormal_ctx"].to(device))
        prompt.eval()

        out = prompt(["object"])
        t_n, t_a = out["object"]
        t_n = F.normalize(t_n.float(), dim=-1)
        t_a = F.normalize(t_a.float(), dim=-1)

        cos_n_anchor = F.cosine_similarity(t_n, anchor_n, dim=0).item()
        cos_a_anchor = F.cosine_similarity(t_a, anchor_a, dim=0).item()
        cos_na = F.cosine_similarity(t_n, t_a, dim=0).item()

        rows.append({
            "lambda_anchor": float(lam),
            "cos_learned_normal_to_normal_anchor": cos_n_anchor,
            "cos_learned_abnormal_to_abnormal_anchor": cos_a_anchor,
            "cos_learned_normal_abnormal": cos_na,
            "semantic_gap_1_minus_normal_abnormal_cos": 1.0 - cos_na,
        })

    df = pd.DataFrame(rows)

    out_dir = Path("paper_results/source_csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "anchor_cosine_analysis.csv"
    df.to_csv(csv_path, index=False)

    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("saved:", csv_path)

    fig_dir = Path("paper_results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5.2, 3.4))
    plt.plot(
        df["lambda_anchor"],
        df["cos_learned_normal_to_normal_anchor"],
        marker="o",
        label="normal prompt to normal anchor",
    )
    plt.plot(
        df["lambda_anchor"],
        df["cos_learned_abnormal_to_abnormal_anchor"],
        marker="o",
        label="abnormal prompt to abnormal anchor",
    )
    plt.plot(
        df["lambda_anchor"],
        df["cos_learned_normal_abnormal"],
        marker="o",
        label="normal to abnormal prompt",
    )
    plt.xlabel(r"$\lambda_{\mathrm{anchor}}$")
    plt.ylabel("Cosine similarity")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7.5)
    plt.tight_layout()

    png_path = fig_dir / "anchor_cosine_analysis.png"
    pdf_path = fig_dir / "anchor_cosine_analysis.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)

    print("saved:", png_path)
    print("saved:", pdf_path)


if __name__ == "__main__":
    main()
