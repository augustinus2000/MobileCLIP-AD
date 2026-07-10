from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

csv_path = Path("paper_results/source_csv/anchor_cosine_analysis.csv")
fig_dir = Path("paper_results/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

plt.figure(figsize=(5.2, 3.2))

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

plt.xlabel(r"$\lambda_{\mathrm{anchor}}$")
plt.ylabel("Cosine similarity")
plt.ylim(0.3, 1.02)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()

png = fig_dir / "anchor_semantic_consistency.png"
pdf = fig_dir / "anchor_semantic_consistency.pdf"

plt.savefig(png, dpi=300)
plt.savefig(pdf)

print("saved:", png)
print("saved:", pdf)
