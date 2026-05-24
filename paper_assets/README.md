# Paper Assets

This folder contains paper-oriented assets generated from the latest semantic knowledge boost experiments.

## Files

- `tables/main_train_table.*`: main train-head comparison table
- `tables/main_concat_table.*`: training-free concat comparison table
- `tables/ablation_core_steps.*`: stepwise core-method ablation
- `tables/ablation_exploratory_variants.*`: exploratory variant comparison
- `figures/method_pipeline.png`: method overview figure
- `figures/best_variant_gain_vs_tac.png`: best-variant gains over TAC
- `writing_outline.md`: paper writing outline

Regenerate with:

```bash
python semantic_knowledge_boost_experiment/generate_paper_assets.py
```