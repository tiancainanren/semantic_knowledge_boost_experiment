# Targeted Boost Summary

- CIFAR-20: `ag_base_g0` train ACC `0.6125`, vs previous best `0.6125` (SAGE-AG[g=0.0]), delta `+0.0000`
- DTD: `ag_texture35_res` train ACC `0.5218`, vs previous best `0.5090` (SAGE-AG[g=0.0]), delta `+0.0128`
- StanfordDogs: `agp_base_g0` train ACC `0.4907`, vs previous best `0.4907` (SAGE-AGP[g=0.0]), delta `+0.0000`
- Flowers: `ag_guidance08_res` train ACC `0.4843`, vs previous best `0.4618` (SAGE-AG[g=0.5]), delta `+0.0225`
- Pets: `agp_residual015_g02` train ACC `0.8256`, vs previous best `0.8217` (SAGE-AGP[g=0.25]), delta `+0.0038`

## KEC Paper Reference Check

- DTD: paper KEC train ACC `0.5130`, our reproduced KEC `0.4830`, our targeted best `0.5218`
- Flowers: paper KEC train ACC `0.7260`, our reproduced KEC `0.4451`, our targeted best `0.4843`
- Pets: paper KEC train ACC `0.6730`, our reproduced KEC `0.7771`, our targeted best `0.8256`