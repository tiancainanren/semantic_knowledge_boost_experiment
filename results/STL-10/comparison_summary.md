# STL-10

## Baselines
- TAC concat ACC: 0.8618, TAC train ACC: 0.9828
- KEC concat ACC: 0.8632, KEC train ACC: 0.9845

## New Methods
- TAC[g=0.0]: concat ACC 0.8618, train ACC 0.9828
- KEC[g=0.5]: concat ACC 0.8632, train ACC 0.9845
- SAGE-A[g=0.5]: concat ACC 0.8635, train ACC 0.9846
- SAGE-AG[g=0.0]: concat ACC 0.8386, train ACC 0.9653
- SAGE-AG[g=0.5]: concat ACC 0.8386, train ACC 0.9855
- SAGE-AGP[g=0.0]: concat ACC 0.8605, train ACC 0.9659
- SAGE-AGP[g=0.25]: concat ACC 0.8605, train ACC 0.9855
- SAGE-AGP[g=0.5]: concat ACC 0.8605, train ACC 0.9850
- SAGE-AGC[adaptive]: concat ACC 0.8386, train ACC 0.9846
- SAGE-AGPC[adaptive]: concat ACC 0.8605, train ACC 0.9854
- SAGE-HYB[adaptive]: concat ACC 0.8599, train ACC 0.9850

- Best concat method: SAGE-A[g=0.5] (0.8635)
- Best train method: SAGE-AG[g=0.5] (0.9855)