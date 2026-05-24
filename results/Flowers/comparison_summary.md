# Flowers

## Baselines
- TAC concat ACC: 0.6775, TAC train ACC: 0.4176
- KEC concat ACC: 0.6833, KEC train ACC: 0.4451

## New Methods
- TAC[g=0.0]: concat ACC 0.6775, train ACC 0.4176
- KEC[g=0.5]: concat ACC 0.6833, train ACC 0.4451
- SAGE-A[g=0.5]: concat ACC 0.6843, train ACC 0.4549
- SAGE-AG[g=0.0]: concat ACC 0.6804, train ACC 0.4049
- SAGE-AG[g=0.5]: concat ACC 0.6804, train ACC 0.4618
- SAGE-AGP[g=0.0]: concat ACC 0.6725, train ACC 0.4098
- SAGE-AGP[g=0.25]: concat ACC 0.6725, train ACC 0.4275
- SAGE-AGP[g=0.5]: concat ACC 0.6725, train ACC 0.4500
- SAGE-AGC[adaptive]: concat ACC 0.6804, train ACC 0.4059
- SAGE-AGPC[adaptive]: concat ACC 0.6725, train ACC 0.4206
- SAGE-HYB[adaptive]: concat ACC 0.6735, train ACC 0.4127

- Best concat method: SAGE-A[g=0.5] (0.6843)
- Best train method: SAGE-AG[g=0.5] (0.4618)