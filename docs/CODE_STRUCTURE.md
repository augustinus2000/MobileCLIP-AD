mobileclip_ad/
├── paper/
│   ├── MobileCLIP-AD.tex
│   └── MobileCLIP-AD.pdf
├── scripts/
│   ├── train_densealign_learnable_prompt.py
│   ├── eval_densealign_learnable_prompt.py
│   ├── eval_raw_mobileclip_stage_text_similarity.py
│   ├── benchmark_lda_efficiency.py
│   ├── benchmark_clip_backbone_efficiency_v2.py
│   ├── benchmark_mobileclip2_variants_objad.py
│   ├── make_final_efficiency_table.py
│   ├── make_table2_sample_std.py
│   ├── analyze_anchor_cosine.py
│   ├── plot_anchor_semantic_consistency.py
│   └── save_abnormal_visualizations.py
├── models/
│   ├── densealign.py
│   ├── learnable_prompt.py
│   ├── mobileclip_extractor.py
│   └── prompts.py
├── datasets/
│   ├── mvtec_dataset.py
│   └── visa_dataset.py
├── docs/
│   ├── PROJECT_FINAL_STATE.md
│   ├── EXPERIMENT_SUMMARY.md
│   ├── RESULT_TABLES.md
│   ├── RUN_COMMANDS.md
│   └── code_structure.txt
├── final_archive/
│   └── mobileclip_ad_final_20260710_163757.tar.gz
├── tools/
└── README.md

Notes:
- outputs/, paper_results/, deprecated/, and logs/ were removed from the active folder after archiving.
- Temporary projection-only pivot and local-refinement comparison scripts were removed from the active scripts/ folder.
- Final checkpoints, result CSVs, historical scripts, and intermediate outputs are preserved inside final_archive/mobileclip_ad_final_20260710_163757.tar.gz.
- The final paper uses LDA with local spatial refinement.
- Projection-only pivot results are deprecated and should not be used as final results.
