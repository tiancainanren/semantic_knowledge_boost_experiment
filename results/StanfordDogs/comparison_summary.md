# StanfordDogs

## Baselines
- TAC concat ACC: 0.4377, TAC train ACC: 0.4580
- KEC concat ACC: 0.4240, KEC train ACC: 0.4365

## New Methods
- TAC[g=0.0]: concat ACC 0.4377, train ACC 0.4580
- KEC[g=0.5]: concat ACC 0.4240, train ACC 0.4365
- SAGE-A[g=0.5]: concat ACC 0.4275, train ACC 0.4400
- SAGE-AG[g=0.0]: concat ACC 0.4336, train ACC 0.4738
- SAGE-AG[g=0.5]: concat ACC 0.4336, train ACC 0.4505
- SAGE-AGP[g=0.0]: concat ACC 0.4400, train ACC 0.4907
- SAGE-AGP[g=0.25]: concat ACC 0.4400, train ACC 0.4641
- SAGE-AGP[g=0.5]: concat ACC 0.4400, train ACC 0.4472
- SAGE-AGC[adaptive]: concat ACC 0.4336, train ACC 0.4505
- SAGE-AGPC[adaptive]: concat ACC 0.4400, train ACC 0.4761
- SAGE-HYB[adaptive]: concat ACC 0.4362, train ACC 0.4583

- Best concat method: SAGE-AGP[g=0.0] (0.4400)
- Best train method: SAGE-AGP[g=0.0] (0.4907)