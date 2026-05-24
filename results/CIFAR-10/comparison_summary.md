# CIFAR-10

## Baselines
- TAC concat ACC: 0.7606, TAC train ACC: 0.9190
- KEC concat ACC: 0.7706, KEC train ACC: 0.8883

## New Methods
- TAC[g=0.0]: concat ACC 0.7606, train ACC 0.9190
- KEC[g=0.5]: concat ACC 0.7706, train ACC 0.8883
- SAGE-A[g=0.5]: concat ACC 0.7672, train ACC 0.8523
- SAGE-AG[g=0.0]: concat ACC 0.7617, train ACC 0.9225
- SAGE-AG[g=0.5]: concat ACC 0.7617, train ACC 0.9069
- SAGE-AGP[g=0.0]: concat ACC 0.7272, train ACC 0.9100
- SAGE-AGP[g=0.25]: concat ACC 0.7272, train ACC 0.9159
- SAGE-AGP[g=0.5]: concat ACC 0.7272, train ACC 0.8033
- SAGE-AGC[adaptive]: concat ACC 0.7617, train ACC 0.9145
- SAGE-AGPC[adaptive]: concat ACC 0.7272, train ACC 0.9131
- SAGE-HYB[adaptive]: concat ACC 0.7690, train ACC 0.9148

- Best concat method: KEC[g=0.5] (0.7706)
- Best train method: SAGE-AG[g=0.0] (0.9225)