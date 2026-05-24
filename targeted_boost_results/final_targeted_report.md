# Final Targeted Boost Report

| Dataset      | Targeted Config     |   Targeted Train ACC | Previous Best Method   |   Previous Best ACC |   Gain vs Previous |   Gain vs TAC | Paper KEC Train ACC   | Our Reproduced KEC ACC   | Targeted - Paper KEC   |
|:-------------|:--------------------|---------------------:|:-----------------------|--------------------:|-------------------:|--------------:|:----------------------|:-------------------------|:-----------------------|
| CIFAR-20     | ag_base_g0          |                61.25 | SAGE-AG[g=0.0]         |               61.25 |               0    |          8.87 | NA                    | NA                       | NA                     |
| DTD          | ag_texture35_res    |                52.18 | SAGE-AG[g=0.0]         |               50.9  |               1.28 |          3.46 | 51.3                  | 48.3                     | 0.88                   |
| StanfordDogs | agp_base_g0         |                49.07 | SAGE-AGP[g=0.0]        |               49.07 |               0    |          3.27 | NA                    | NA                       | NA                     |
| Flowers      | ag_guidance08_res   |                48.43 | SAGE-AG[g=0.5]         |               46.18 |               2.25 |          6.67 | 72.6                  | 44.51                    | -24.17                 |
| Pets         | agp_residual015_g02 |                82.56 | SAGE-AGP[g=0.25]       |               82.17 |               0.38 |          5.21 | 67.3                  | 77.71                    | 15.26                  |

## Notes

- This targeted round is a post-hoc dataset-focused enhancement study on the most promising datasets.
- `CIFAR-20` and `StanfordDogs` were already close to the current method ceiling under this semantic family; the extra search did not improve them further.
- `DTD`, `Flowers`, and `Pets` gained further improvements after targeted tuning.
- Paper KEC references are only available for the datasets explicitly listed in the extracted PDF tables; `CIFAR-20` and `StanfordDogs` were not found in the reported tables we extracted.