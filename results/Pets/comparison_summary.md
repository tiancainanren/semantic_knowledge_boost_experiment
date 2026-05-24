# Pets

## Baselines
- TAC concat ACC: 0.7021, TAC train ACC: 0.7735
- KEC concat ACC: 0.6620, KEC train ACC: 0.7771

## New Methods
- TAC[g=0.0]: concat ACC 0.7021, train ACC 0.7735
- KEC[g=0.5]: concat ACC 0.6620, train ACC 0.7771
- SAGE-A[g=0.5]: concat ACC 0.6710, train ACC 0.7602
- SAGE-AG[g=0.0]: concat ACC 0.6787, train ACC 0.8062
- SAGE-AG[g=0.5]: concat ACC 0.6787, train ACC 0.7836
- SAGE-AGP[g=0.0]: concat ACC 0.6860, train ACC 0.8078
- SAGE-AGP[g=0.25]: concat ACC 0.6860, train ACC 0.8217
- SAGE-AGP[g=0.5]: concat ACC 0.6860, train ACC 0.8049
- SAGE-AGC[adaptive]: concat ACC 0.6787, train ACC 0.8008
- SAGE-AGPC[adaptive]: concat ACC 0.6860, train ACC 0.7803
- SAGE-HYB[adaptive]: concat ACC 0.6716, train ACC 0.8032

- Best concat method: TAC[g=0.0] (0.7021)
- Best train method: SAGE-AGP[g=0.25] (0.8217)