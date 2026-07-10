# MobileCLIP-AD Experiment Summary

## Summary

MobileCLIP-AD is a source-supervised cross-dataset industrial anomaly segmentation framework based on frozen MobileCLIP2.

The final project investigates whether MobileCLIP2 hierarchical features can be adapted for dense anomaly localization through lightweight dense adaptation and object-agnostic text-side adaptation.

## Final method

The final MobileCLIP-AD pipeline uses:

- Frozen MobileCLIP2-S3
- Stage-2 and stage-3 hierarchical features
- Lightweight Dense Adaptation, LDA
  - dense projection
  - local spatial refinement
  - learnable stage fusion
- Object-agnostic learnable prompts
  - normal object
  - defective object
- Object-agnostic prototype anchoring

## Why dense adaptation is needed

MobileCLIP2 is trained as an image-level vision-language model. Its hierarchical stage features are not directly aligned with text embeddings for dense pixel-level similarity.

Raw stage-3 text similarity performs poorly. Dense projection provides the major performance improvement. LDA adapts MobileCLIP2 hierarchical features into dense text-comparable features.

## Final default setting

- Backbone: MobileCLIP2-S3
- Input size: 256
- Stages: 2 and 3
- Align mode: local
- Prompt: object-agnostic learnable prompt
- Anchor: object-agnostic prototype anchor
- Anchor weight: 0.5
- Source supervision: anomaly masks from source dataset
- Main setting: VisA -> MVTec AD

## Final conclusion

MobileCLIP-AD is not a strict SOTA accuracy method.

The final conclusion is:

- MobileCLIP2-S3 can be adapted for dense industrial anomaly segmentation.
- The method provides competitive Pixel AUROC under several cross-dataset settings.
- ViT-L/14 references generally provide stronger Pixel AP, F1, and IoU.
- MobileCLIP2-S3 provides lower image-encoder FLOPs and a practical accuracy-efficiency trade-off.
- Pixel AP and confidence calibration remain limitations.

## Abandoned projection-only pivot

A projection-only version was considered after observing that local refinement did not always provide a large standalone gain. However, the projection-only pivot was abandoned because it made the existing paper and result tables inconsistent and produced unstable stage-ablation behavior.

The final archived project therefore uses the previous LDA-based pipeline with local refinement.

## Final archived file

final_archive/mobileclip_ad_final_20260710_163757.tar.gz
