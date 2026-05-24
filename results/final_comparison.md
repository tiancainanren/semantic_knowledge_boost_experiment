# Final Comparison

## CIFAR-10

- Best concat: `KEC[g=0.5]` ACC `0.7706`
- Delta vs TAC concat: `+0.0100`
- Delta vs KEC concat: `+0.0000`
- Best train: `SAGE-AG[g=0.0]` ACC `0.9225`
- Delta vs TAC train: `+0.0035`
- Delta vs KEC train: `+0.0342`

## CIFAR-20

- Best concat: `KEC[g=0.5]` ACC `0.5747`
- Delta vs TAC concat: `+0.0304`
- Delta vs KEC concat: `+0.0000`
- Best train: `SAGE-AG[g=0.0]` ACC `0.6125`
- Delta vs TAC train: `+0.0887`
- Delta vs KEC train: `+0.0850`

## DTD

- Best concat: `SAGE-AG[g=0.0]` ACC `0.4793`
- Delta vs TAC concat: `+0.0223`
- Delta vs KEC concat: `+0.0351`
- Best train: `SAGE-AG[g=0.0]` ACC `0.5090`
- Delta vs TAC train: `+0.0218`
- Delta vs KEC train: `+0.0261`

## Flowers

- Best concat: `SAGE-A[g=0.5]` ACC `0.6843`
- Delta vs TAC concat: `+0.0069`
- Delta vs KEC concat: `+0.0010`
- Best train: `SAGE-AG[g=0.5]` ACC `0.4618`
- Delta vs TAC train: `+0.0441`
- Delta vs KEC train: `+0.0167`

## Pets

- Best concat: `TAC[g=0.0]` ACC `0.7021`
- Delta vs TAC concat: `+0.0000`
- Delta vs KEC concat: `+0.0401`
- Best train: `SAGE-AGP[g=0.25]` ACC `0.8217`
- Delta vs TAC train: `+0.0482`
- Delta vs KEC train: `+0.0447`

## STL-10

- Best concat: `SAGE-A[g=0.5]` ACC `0.8635`
- Delta vs TAC concat: `+0.0018`
- Delta vs KEC concat: `+0.0003`
- Best train: `SAGE-AG[g=0.5]` ACC `0.9855`
- Delta vs TAC train: `+0.0028`
- Delta vs KEC train: `+0.0010`

## StanfordDogs

- Best concat: `SAGE-AGP[g=0.0]` ACC `0.4400`
- Delta vs TAC concat: `+0.0023`
- Delta vs KEC concat: `+0.0160`
- Best train: `SAGE-AGP[g=0.0]` ACC `0.4907`
- Delta vs TAC train: `+0.0327`
- Delta vs KEC train: `+0.0542`

## Family Averages vs TAC

- `SAGE-AG`: mean concat delta `-0.0000`, mean train delta `+0.0129`
- `SAGE-AGP`: mean concat delta `-0.0019`, mean train delta `+0.0055`
- `SAGE-HYB`: mean concat delta `-0.0114`, mean train delta `+0.0021`
- `SAGE-AGPC`: mean concat delta `-0.0019`, mean train delta `+0.0019`
- `TAC`: mean concat delta `+0.0000`, mean train delta `+0.0000`
- `SAGE-AGC`: mean concat delta `-0.0000`, mean train delta `-0.0007`
- `KEC`: mean concat delta `-0.0027`, mean train delta `-0.0029`
- `SAGE-A`: mean concat delta `+0.0007`, mean train delta `-0.0068`