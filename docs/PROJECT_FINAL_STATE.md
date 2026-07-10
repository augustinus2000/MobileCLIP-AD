# MobileCLIP-AD Final State

## Final status

This project is finalized and archived.

The final paper version uses the LDA-based MobileCLIP-AD pipeline with local spatial refinement.

Final archive:

final_archive/mobileclip_ad_final_20260710_163757.tar.gz

The project should now be treated as a completed archived research draft, not an active experimental branch.

## Final paper title

Exploring Lightweight Dense Adaptation of MobileCLIP for Cross-Dataset Industrial Anomaly Segmentation

## Final research positioning

This work is not intended as a strict SOTA anomaly segmentation method.

The final positioning is:

- empirical study
- lightweight dense adaptation
- MobileCLIP2-S3 as a mobile-oriented vision-language backbone
- cross-dataset industrial anomaly segmentation
- accuracy-efficiency trade-off compared with ViT-L/14-based CLIP references

Recommended claim:

MobileCLIP2 hierarchical features can be adapted for dense industrial anomaly segmentation using lightweight dense adaptation and object-agnostic text-side adaptation. The method provides a practical accuracy-efficiency trade-off compared with larger ViT-L/14-based CLIP references.

Avoid overclaiming:

- Do not claim strict SOTA performance.
- Do not claim local refinement consistently improves all metrics.
- Do not claim MobileCLIP2-S3 is lighter than every compact CLIP backbone.
- Do not claim this is a general zero-shot anomaly detection method.

## Final method

Final MobileCLIP-AD consists of:

1. Frozen MobileCLIP2-S3 backbone
2. Stage-2 and stage-3 hierarchical feature extraction
3. Lightweight Dense Adaptation, LDA
   - dense projection into the text embedding dimension
   - lightweight local spatial refinement
   - learnable stage fusion
4. Object-agnostic learnable prompt adaptation
   - normal object
   - defective object
5. Object-agnostic prototype anchoring
6. Source-supervised cross-dataset anomaly segmentation

## Final default setting

- Backbone: MobileCLIP2-S3
- Pretrained weight: dfndr2b
- Input size: 256
- Stages: 2 and 3
- Align mode: local
- Prompt: object-agnostic learnable prompt
- Object word: object
- Anchor: object-agnostic anchor
- Anchor weight: 0.5
- Prompt length: n_ctx = 4
- Training: source-supervised with anomaly masks
- Main transfer: VisA -> MVTec AD
- Additional transfers: MVTec -> VisA, VisA -> BTAD, VisA -> MPDD, VisA -> DAGM, VisA -> KSDD2

## Final checkpoints

VisA-trained final checkpoints:

- outputs/lda_s3_objprompt_objanchor_seed0_visa
- outputs/lda_s3_objprompt_objanchor_seed1_visa
- outputs/lda_s3_objprompt_objanchor_seed2_visa

MVTec-trained final checkpoints:

- outputs/lda_s3_objprompt_objanchor_seed0_mvtec
- outputs/lda_s3_objprompt_objanchor_seed1_mvtec
- outputs/lda_s3_objprompt_objanchor_seed2_mvtec

These checkpoints are preserved inside the final archive.

## Important abandoned direction

A projection-only final pivot was considered and then abandoned.

Reason:

- It made existing tables inconsistent.
- Projection-only stage ablation behaved unexpectedly and produced unstable or very low results.
- It weakened the method novelty further.
- The previous LDA-based pipeline is more coherent as the final archived project.

Projection-only outputs should be treated as deprecated and should not be used for the final paper.

## Final paper status

The final paper file is:

- paper/MobileCLIP-AD.tex
- paper/MobileCLIP-AD.pdf

The qualitative local-refinement comparison figure was removed because local-refinement behavior was dataset-dependent and not central to the final conservative claim.

## Build command

cd ~/MobileCLIP/mobileclip_ad/paper
latexmk -g -pdf MobileCLIP-AD.tex
grep -n "not found|No file|undefined|Citation.*undefined|Reference.*undefined|Float too large" MobileCLIP-AD.log || true

## Final active-folder cleanup

After final archiving, the active working folder was cleaned.

Removed or moved from the active folder:

- outputs/
- paper_results/
- deprecated/
- logs/
- temporary projection-only pivot scripts
- local-refinement comparison/contact-sheet scripts
- obsolete qualitative figure generation scripts
- old root-level run_*.sh scripts

The active scripts/ directory now keeps only the minimal scripts needed for final reproduction, evaluation, efficiency checks, and limited visualization.

Full historical scripts, checkpoints, result CSVs, and intermediate outputs are preserved inside the final archive:

final_archive/mobileclip_ad_final_20260710_163757.tar.gz

