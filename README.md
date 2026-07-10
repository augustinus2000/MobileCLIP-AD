# MobileCLIP-AD

> **Personal Research Project (2026)**  
> Exploring Lightweight Dense Adaptation of MobileCLIP2 for Cross-Dataset Industrial Anomaly Segmentation

---

## Overview

MobileCLIP-AD is a personal research project that investigates whether **MobileCLIP2**, a lightweight vision-language model designed for mobile deployment, can be adapted for dense industrial anomaly segmentation.

Unlike conventional CLIP-based anomaly detection methods that rely on large ViT backbones, this project explores a lightweight alternative while maintaining competitive localization performance under cross-dataset transfer settings.

---

## Motivation

Most recent CLIP-based anomaly segmentation methods are built upon large Vision Transformers (e.g., CLIP ViT-L/14).

Although they achieve strong performance, they require considerable computational resources and are less suitable for mobile or resource-constrained environments.

This project explores the following question:

> **Can MobileCLIP2 hierarchical features be effectively adapted for dense anomaly localization without fine-tuning the backbone?**

---

## Proposed Method

The proposed MobileCLIP-AD framework consists of:

- Frozen MobileCLIP2-S3 backbone
- Hierarchical Stage-2 / Stage-3 feature extraction
- Lightweight Dense Adaptation (LDA)
  - Dense projection
  - Local spatial refinement
  - Learnable stage fusion
- Object-agnostic learnable prompts
- Object-agnostic prototype anchoring
- Cross-dataset supervised anomaly localization

---

## Key Features

- Lightweight MobileCLIP2 backbone
- Dense feature adaptation instead of backbone fine-tuning
- Learnable object-agnostic prompts
- Prototype anchor regularization
- Cross-dataset evaluation
- Mobile-oriented efficiency analysis

---

## Experimental Settings

Source datasets

- VisA
- MVTec AD

Target datasets

- MVTec AD
- VisA
- BTAD
- MPDD
- DAGM
- KSDD2

Evaluation metrics

- Pixel AUROC
- Pixel AP
- Pixel F1
- Pixel IoU
- AUPRO
- Image AUROC
- Image AP
- Image F1

---

## Main Findings

The study shows that:

- MobileCLIP2 hierarchical features are not directly suitable for dense text matching.
- Lightweight Dense Adaptation substantially improves dense anomaly localization.
- Object-agnostic prompt learning further improves localization quality.
- MobileCLIP2 provides an attractive accuracy-efficiency trade-off compared with larger CLIP ViT-L/14 models.

---

## Repository Structure

```
mobileclip_ad/
├── paper/
├── scripts/
├── models/
├── datasets/
├── docs/
├── tools/
└── README.md
```

Detailed documentation is available in:

- docs/PROJECT_FINAL_STATE.md
- docs/EXPERIMENT_SUMMARY.md
- docs/RESULT_TABLES.md
- docs/RUN_COMMANDS.md

---

## Project Status

This project has been finalized and archived.

The repository is preserved as a research portfolio and implementation reference.

---

## Author

Junhyeok Im

Graduate Researcher  
School of Electronics and Electrical Engineering  
Kyungpook National University

---

## Copyright

Copyright © 2026 Junhyeok Im

This repository is provided for research and portfolio purposes only.

Unauthorized reproduction, redistribution, or commercial use is prohibited.