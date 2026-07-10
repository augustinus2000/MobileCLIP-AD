# Final Result Tables

This document records the final paper-level result tables for MobileCLIP-AD.

## Final setting

- Method: MobileCLIP-AD
- Backbone: frozen MobileCLIP2-S3
- Adaptation: LDA with local spatial refinement
- Prompt: object-agnostic learnable prompt
- Anchor: object-agnostic prototype anchor
- Default stages: stage 2 + stage 3
- Default transfer: VisA -> MVTec AD
- Metrics are reported in percentage.
- Results are reported as mean plus/minus standard deviation over three seeds.

## Stage selection analysis

VisA -> MVTec AD, MobileCLIP2-S3, LDA.

Stage 2+3 is the default configuration.

Key result:

- Stage 2+3 default: Pixel AUROC 89.79 +/- 0.32, Pixel AP 33.51 +/- 0.54, AUPRO 82.88 +/- 0.26, Pixel F1 38.41 +/- 0.33, Pixel IoU 24.67 +/- 0.21, Image AUROC 87.26 +/- 1.01, Image AP 93.99 +/- 0.37, Image F1 90.97 +/- 0.44.

Rationale:

- Stage 2+3 provides the best overall trade-off.
- Stage 1+2+3 improves several pixel-level metrics but slightly degrades image-level performance.
- Stage 3 gives strong image-level performance but weaker localization detail.
- Stage 1 alone lacks semantic abstraction.

## Main component analysis

Final selected method:

Object-agnostic prompt + object-agnostic anchor + LDA.

Final selected result under VisA -> MVTec AD:

- Pixel AUROC: 89.79 +/- 0.26
- Pixel AP: 33.51 +/- 0.44
- AUPRO: 82.88 +/- 0.21
- Pixel F1: 38.41 +/- 0.27
- Pixel IoU: 24.67 +/- 0.17
- Image AUROC: 87.26 +/- 0.83
- Image AP: 93.99 +/- 0.30
- Image F1: 90.97 +/- 0.36

## Necessity of dense adaptation

VisA -> MVTec AD, MobileCLIP2-S3.

- Raw stage-3 text similarity, best direct score: Pixel AUROC 65.58, Pixel AP 6.80.
- Projection only + fixed prompts: Pixel AUROC 89.57 +/- 0.02, Pixel AP 29.92 +/- 0.18.
- Projection + local refinement + fixed prompts: Pixel AUROC 89.22 +/- 0.27, Pixel AP 29.52 +/- 0.66.
- Projection + prompt + anchor: Pixel AUROC 90.03 +/- 0.21, Pixel AP 33.12 +/- 0.75.
- Full MobileCLIP-AD: Pixel AUROC 89.79 +/- 0.32, Pixel AP 33.51 +/- 0.54.

Interpretation:

- Raw MobileCLIP2 stage features are not directly suitable for dense text matching.
- Dense projection is the largest source of improvement.
- Local refinement is included in final LDA but should not be overclaimed as a large standalone gain.
- Prompt adaptation and anchoring improve localization and calibration trade-off.

## Cross-dataset result summary

The final paper reports:

- VisA -> MVTec AD
- MVTec AD -> VisA
- VisA -> BTAD
- VisA -> MPDD
- VisA -> DAGM
- VisA -> KSDD2

Compared with ViT-L/14 references, MobileCLIP2-S3 generally has lower Pixel AP, F1, and IoU but competitive Pixel AUROC and several strong image-level results.

The intended claim is an accuracy-efficiency trade-off, not SOTA accuracy.

## Efficiency summary

MobileCLIP2-S3 at 256 input:

- Total params: 248.89M
- Image params: 125.24M
- Text params: 123.65M
- LDA head: 3.26M
- Prompt: 0.01M
- Trainable: 3.27M
- Image FLOPs: 29.68G

Compared with CLIP ViT-L/14 at 224:

- Image encoder parameter reduction: 2.43x
- Image encoder FLOPs reduction: 3.93x

Compared with CLIP ViT-L/14 at 336:

- Image encoder FLOPs reduction: 8.83x
