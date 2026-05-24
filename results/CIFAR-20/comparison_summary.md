# CIFAR-20

## Baselines
- TAC concat ACC: 0.5443, TAC train ACC: 0.5238
- KEC concat ACC: 0.5747, KEC train ACC: 0.5275

## New Methods
- TAC[g=0.0]: concat ACC 0.5443, train ACC 0.5238
- KEC[g=0.5]: concat ACC 0.5747, train ACC 0.5275
- SAGE-A[g=0.5]: concat ACC 0.5732, train ACC 0.5508
- SAGE-AG[g=0.0]: concat ACC 0.5686, train ACC 0.6125
- SAGE-AG[g=0.5]: concat ACC 0.5686, train ACC 0.5141
- SAGE-AGP[g=0.0]: concat ACC 0.5677, train ACC 0.5420
- SAGE-AGP[g=0.25]: concat ACC 0.5677, train ACC 0.5091
- SAGE-AGP[g=0.5]: concat ACC 0.5677, train ACC 0.5588
- SAGE-AGC[adaptive]: concat ACC 0.5686, train ACC 0.5080
- SAGE-AGPC[adaptive]: concat ACC 0.5677, train ACC 0.5044
- SAGE-HYB[adaptive]: concat ACC 0.4887, train ACC 0.5184

- Best concat method: KEC[g=0.5] (0.5747)
- Best train method: SAGE-AG[g=0.0] (0.6125)