# DTD

## Baselines
- TAC concat ACC: 0.4569, TAC train ACC: 0.4872
- KEC concat ACC: 0.4441, KEC train ACC: 0.4830

## New Methods
- TAC[g=0.0]: concat ACC 0.4569, train ACC 0.4872
- KEC[g=0.5]: concat ACC 0.4441, train ACC 0.4830
- SAGE-A[g=0.5]: concat ACC 0.4590, train ACC 0.4713
- SAGE-AG[g=0.0]: concat ACC 0.4793, train ACC 0.5090
- SAGE-AG[g=0.5]: concat ACC 0.4793, train ACC 0.5074
- SAGE-AGP[g=0.0]: concat ACC 0.4734, train ACC 0.5059
- SAGE-AGP[g=0.25]: concat ACC 0.4734, train ACC 0.4947
- SAGE-AGP[g=0.5]: concat ACC 0.4734, train ACC 0.5011
- SAGE-AGC[adaptive]: concat ACC 0.4793, train ACC 0.4926
- SAGE-AGPC[adaptive]: concat ACC 0.4734, train ACC 0.4952
- SAGE-HYB[adaptive]: concat ACC 0.4622, train ACC 0.4840

- Best concat method: SAGE-AG[g=0.0] (0.4793)
- Best train method: SAGE-AG[g=0.0] (0.5090)