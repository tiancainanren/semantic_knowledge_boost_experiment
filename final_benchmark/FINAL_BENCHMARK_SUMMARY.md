# Final Benchmark Comparison

## Reading Guide
- `clip_baseline_acc`: image-only CLIP embedding + spherical KMeans.
- `tac_local_acc`: our local TAC baseline. If an official TAC reproduction package exists, it is preferred; otherwise the TAC branch inside `semantic_knowledge_boost_experiment` is used.
- `kec_local_acc`: our local KEC reproduction from `kec_paper_reproduction`.
- `tac_paper_acc` / `kec_paper_acc`: values explicitly reported in the corresponding papers and already extracted into local reference files.
- `ours_acc`: our best result for that protocol. No-train uses the best concat result; train uses the best targeted train result when available, otherwise the best train result from the main semantic experiment.
- Positive `gain_vs_best_local_prior` means our method beats every available local baseline on that dataset under the same protocol.
- Positive `gain_vs_best_paper_prior` means our method exceeds the best paper-reported TAC/KEC value available in this repo for that dataset and protocol.

## No-Train Comparison

| Dataset | CLIP | TAC Local | TAC Paper | KEC Local | KEC Paper | Ours | Ours Method | Gain vs Best Local | Gain vs Best Paper |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| CIFAR-10 | 0.7621 | 0.8931 | 0.8810 | 0.6774 | NA | 0.7690 | SAGE-HYB[adaptive] | -0.1241 | -0.1120 |
| CIFAR-20 | NA | 0.5443 | NA | 0.5678 | NA | 0.5732 | SAGE-A[g=0.5] | 0.0054 | NA |
| DTD | 0.4399 | 0.4375 | 0.4790 | 0.4654 | 0.4740 | 0.4793 | SAGE-AG[g=0.0] | 0.0138 | 0.0003 |
| Flowers | 0.5725 | 0.6775 | NA | 0.6873 | 0.7280 | 0.6843 | SAGE-A[g=0.5] | -0.0029 | -0.0437 |
| Pets | 0.5168 | 0.7021 | NA | 0.6836 | 0.6780 | 0.6860 | SAGE-AGP[g=0.25] | -0.0161 | 0.0080 |
| STL-10 | 0.9625 | 0.8326 | 0.8190 | 0.8622 | NA | 0.8635 | SAGE-A[g=0.5] | -0.0990 | 0.0445 |
| StanfordDogs | 0.2066 | 0.4377 | NA | 0.5048 | NA | 0.5071 | second_wave:dual_cross_preserve | 0.0023 | NA |

## Train Comparison

| Dataset | TAC Local | TAC Paper | KEC Local | KEC Paper | Ours | Ours Method | Gain vs Best Local | Gain vs Best Paper |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| CIFAR-10 | 0.9143 | 0.9630 | 0.9062 | NA | 0.9225 | SAGE-AG[g=0.0] | 0.0082 | -0.0405 |
| CIFAR-20 | 0.5238 | NA | 0.5540 | NA | 0.6125 | ag_base_g0 | 0.0585 | NA |
| DTD | 0.4979 | 0.5330 | 0.4798 | 0.5130 | 0.5218 | ag_texture35_res | 0.0239 | -0.0112 |
| Flowers | 0.4176 | NA | 0.4333 | 0.7260 | 0.4843 | ag_guidance08_res | 0.0510 | -0.2417 |
| Pets | 0.7735 | NA | 0.7768 | 0.6730 | 0.8256 | agp_residual015_g02 | 0.0488 | 0.1526 |
| STL-10 | 0.9719 | 0.9500 | 0.9844 | NA | 0.9855 | SAGE-AG[g=0.5] | 0.0011 | 0.0355 |
| StanfordDogs | 0.4580 | NA | 0.5374 | NA | 0.5651 | second_wave:dual_cross_preserve | 0.0277 | NA |

## Takeaways
- On the no-train protocol, our method beats the strongest available local prior baseline on 3/7 datasets.
- On the train protocol, our method beats the strongest available local prior baseline on 7/7 datasets.
- Datasets with missing paper values should be interpreted only against local baselines, not as a strict paper-level comparison.
- The KEC branch here is an offline local surrogate of LLM-generated knowledge, so remaining gaps versus the KEC paper should be read with that caveat in mind.