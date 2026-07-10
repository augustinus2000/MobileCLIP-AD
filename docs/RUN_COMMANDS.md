# Final Run Commands

This document records the final commands associated with the archived MobileCLIP-AD project.

The active project folder has been cleaned. Final checkpoints and results are preserved inside:

final_archive/mobileclip_ad_final_20260710_163757.tar.gz

Use the commands below only if the archive is extracted or the original outputs are restored.

## Build final paper

cd ~/MobileCLIP/mobileclip_ad/paper
latexmk -g -pdf MobileCLIP-AD.tex
grep -n "not found|No file|undefined|Citation.*undefined|Reference.*undefined|Float too large" MobileCLIP-AD.log || true

## Check final archive

cd ~/MobileCLIP/mobileclip_ad
ls -lh final_archive/mobileclip_ad_final_20260710_163757.tar.gz
tar -tzf final_archive/mobileclip_ad_final_20260710_163757.tar.gz | head -80

## Extract final archive if needed

cd ~/MobileCLIP/mobileclip_ad
mkdir -p restored_final
tar -xzf final_archive/mobileclip_ad_final_20260710_163757.tar.gz -C restored_final

## Final training command pattern

Final method:

- train_dataset: visa or mvtec
- align_mode: local
- stages: 2 3
- object_agnostic_prompt enabled
- object_agnostic_anchor enabled
- lambda_prompt_anchor: 0.5
- n_ctx: 4

Example VisA training:

cd ~/MobileCLIP/mobileclip_ad

python scripts/train_densealign_learnable_prompt.py \
  --train_dataset visa \
  --visa_root ../datasets/visa \
  --visa_split_csv ../datasets/visa/split_csv/1cls.csv \
  --mvtec_root ../datasets/mvtec_anomaly_detection \
  --model_name MobileCLIP2-S3 \
  --pretrained dfndr2b \
  --image_size 256 \
  --stages 2 3 \
  --align_mode local \
  --learnable_prompt \
  --object_agnostic_prompt \
  --object_word object \
  --object_agnostic_anchor \
  --lambda_prompt_anchor 0.5 \
  --lambda_prompt_reg 0.001 \
  --lambda_focal 1.0 \
  --lambda_dice 1.0 \
  --lambda_img 0.5 \
  --lambda_cons 0.05 \
  --topk_ratio 0.01 \
  --mask_dilate 5 \
  --epochs 1 \
  --batch_size 32 \
  --num_workers 4 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --seed 0 \
  --output_dir outputs/lda_s3_objprompt_objanchor_seed0_visa

## Final evaluation command pattern

Example VisA -> MVTec evaluation:

cd ~/MobileCLIP/mobileclip_ad

python scripts/eval_densealign_learnable_prompt.py \
  --eval_dataset mvtec \
  --mvtec_root ../datasets/mvtec_anomaly_detection \
  --ckpt outputs/lda_s3_objprompt_objanchor_seed0_visa/best.pth \
  --output_dir paper_results/source_csv/final_visa_to_mvtec_seed0 \
  --batch_size 32 \
  --num_workers 4 \
  --aupro_thresholds 200

## Final checkpoint names

VisA-trained:

outputs/lda_s3_objprompt_objanchor_seed0_visa
outputs/lda_s3_objprompt_objanchor_seed1_visa
outputs/lda_s3_objprompt_objanchor_seed2_visa

MVTec-trained:

outputs/lda_s3_objprompt_objanchor_seed0_mvtec
outputs/lda_s3_objprompt_objanchor_seed1_mvtec
outputs/lda_s3_objprompt_objanchor_seed2_mvtec

Stage ablation:

outputs/stage_ablation_final_all_s1_seed*
outputs/stage_ablation_final_all_s2_seed*
outputs/stage_ablation_final_all_s3_seed*
outputs/stage_ablation_final_all_s12_seed*
outputs/stage_ablation_final_all_s13_seed*
outputs/stage_ablation_final_all_s23_seed*
outputs/stage_ablation_final_all_s123_seed*

## Deprecated direction

Do not use projection-only pivot results as final results.

Projection-only finalization was attempted and abandoned.
Final paper uses LDA with local spatial refinement.
