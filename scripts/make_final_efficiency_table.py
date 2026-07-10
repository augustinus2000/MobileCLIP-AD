from pathlib import Path
import pandas as pd

base = Path("paper_results/source_csv")
base.mkdir(parents=True, exist_ok=True)

rows = [
    {
        "Backbone": "CLIP ViT-B/32",
        "Input": "224",
        "Total Params (M)": 151.28,
        "Visual Params (M)": 87.85,
        "Text Params (M)": 63.43,
        "Visual GMACs": 3.30,
        "Visual GFLOPs": 6.60,
        "Latency bs=1 (ms)": 2.61,
    },
    {
        "Backbone": "CLIP ViT-B/16",
        "Input": "224",
        "Total Params (M)": 149.62,
        "Visual Params (M)": 86.19,
        "Text Params (M)": 63.43,
        "Visual GMACs": 12.66,
        "Visual GFLOPs": 25.33,
        "Latency bs=1 (ms)": 4.11,
    },
    {
        "Backbone": "CLIP ViT-L/14",
        "Input": "224",
        "Total Params (M)": 427.62,
        "Visual Params (M)": 303.97,
        "Text Params (M)": 123.65,
        "Visual GMACs": 58.36,
        "Visual GFLOPs": 116.73,
        "Latency bs=1 (ms)": 15.18,
    },
    {
        "Backbone": "CLIP ViT-L/14",
        "Input": "336",
        "Total Params (M)": 427.94,
        "Visual Params (M)": 304.29,
        "Text Params (M)": 123.65,
        "Visual GMACs": 131.03,
        "Visual GFLOPs": 262.07,
        "Latency bs=1 (ms)": 29.64,
    },
    {
        "Backbone": "MobileCLIP2-S3",
        "Input": "256",
        "Total Params (M)": 248.89,
        "Visual Params (M)": 125.24,
        "Text Params (M)": 123.65,
        "Visual GMACs": 14.84,
        "Visual GFLOPs": 29.68,
        "Latency bs=1 (ms)": 14.65,
    },
    {
        "Backbone": "MobileCLIP2-S4",
        "Input": "256",
        "Total Params (M)": 445.44,
        "Visual Params (M)": 321.78,
        "Text Params (M)": 123.65,
        "Visual GMACs": 27.79,
        "Visual GFLOPs": 55.59,
        "Latency bs=1 (ms)": 15.32,
    },
]

df = pd.DataFrame(rows)

# Add ratios against MobileCLIP2-S3
s3 = df[df["Backbone"].eq("MobileCLIP2-S3")].iloc[0]
df["Visual Params / S3"] = df["Visual Params (M)"] / s3["Visual Params (M)"]
df["GFLOPs / S3"] = df["Visual GFLOPs"] / s3["Visual GFLOPs"]
df["Latency / S3"] = df["Latency bs=1 (ms)"] / s3["Latency bs=1 (ms)"]

numeric_path = base / "final_backbone_efficiency_numeric.csv"
paper_path = base / "final_backbone_efficiency_paper.csv"

df.to_csv(numeric_path, index=False)

paper = df.copy()
for c in [
    "Total Params (M)",
    "Visual Params (M)",
    "Text Params (M)",
    "Visual GMACs",
    "Visual GFLOPs",
    "Latency bs=1 (ms)",
    "Visual Params / S3",
    "GFLOPs / S3",
    "Latency / S3",
]:
    paper[c] = paper[c].map(lambda x: f"{x:.2f}")

paper.to_csv(paper_path, index=False)

print("===== NUMERIC =====")
print(df.to_string(index=False))

print("\n===== PAPER =====")
print(paper.to_string(index=False))

print("\nsaved:", numeric_path)
print("saved:", paper_path)
