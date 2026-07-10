from pathlib import Path
import pandas as pd
import numpy as np

base = Path("paper_results/source_csv")

rows = [
    ("MobileCLIP2-S3", "VisA$\\rightarrow$MVTec", "lda_visa_to_mvtec_s3_seed{}", "mvtec"),
    ("CLIP ViT-L/14@224", "VisA$\\rightarrow$MVTec", "lda_visa_to_mvtec_vitl14_224_seed{}", "mvtec"),
    ("CLIP ViT-L/14@336", "VisA$\\rightarrow$MVTec", "lda_visa_to_mvtec_vitl14_336_seed{}", "mvtec"),

    ("MobileCLIP2-S3", "MVTec$\\rightarrow$VisA", "lda_mvtec_to_visa_s3_seed{}", "visa"),
    ("CLIP ViT-L/14@224", "MVTec$\\rightarrow$VisA", "lda_mvtec_to_visa_vitl14_224_seed{}", "visa"),
    ("CLIP ViT-L/14@336", "MVTec$\\rightarrow$VisA", "lda_mvtec_to_visa_vitl14_336_seed{}", "visa"),

    ("MobileCLIP2-S3", "VisA$\\rightarrow$BTAD", "lda_visa_to_btad_s3_seed{}", "btad"),
    ("CLIP ViT-L/14@224", "VisA$\\rightarrow$BTAD", "lda_visa_to_btad_vitl14_224_seed{}", "btad"),
    ("CLIP ViT-L/14@336", "VisA$\\rightarrow$BTAD", "lda_visa_to_btad_vitl14_336_seed{}", "btad"),

    ("MobileCLIP2-S3", "VisA$\\rightarrow$MPDD", "lda_visa_to_mpdd_s3_seed{}", "mpdd"),
    ("CLIP ViT-L/14@224", "VisA$\\rightarrow$MPDD", "lda_visa_to_mpdd_vitl14_224_seed{}", "mpdd"),
    ("CLIP ViT-L/14@336", "VisA$\\rightarrow$MPDD", "lda_visa_to_mpdd_vitl14_336_seed{}", "mpdd"),
]

cols = [
    "macro_pixel_auroc", "macro_pixel_ap", "macro_pixel_aupro",
    "macro_pixel_f1", "macro_pixel_iou",
    "macro_image_auroc", "macro_image_ap", "macro_image_f1",
]

out = []

for backbone, task, pattern, ds in rows:
    vals = []

    for seed in [0, 1, 2]:
        p = base / pattern.format(seed) / f"{ds}_summary_metrics.csv"
        if not p.exists():
            raise FileNotFoundError(p)

        df = pd.read_csv(p)
        row = df[df["score_name"] == "score_mean"].iloc[0]
        vals.append(row[cols].to_numpy(dtype=float) * 100)

    vals = np.stack(vals, axis=0)
    mean = vals.mean(axis=0)
    std = vals.std(axis=0, ddof=1)

    formatted = [f"{m:.2f}$\\pm${s:.2f}" for m, s in zip(mean, std)]
    out.append([backbone, task] + formatted)

columns = [
    "Backbone", "Task",
    "Pixel AUROC", "Pixel AP", "AUPRO", "Pixel F1", "Pixel IoU",
    "Image AUROC", "Image AP", "Image F1"
]

df_out = pd.DataFrame(out, columns=columns)

out_csv = base / "lda_cross_dataset_3seed_summary_scoremean_sample_std.csv"
df_out.to_csv(out_csv, index=False)

print("saved:", out_csv)
print()
print(df_out.to_string(index=False))
