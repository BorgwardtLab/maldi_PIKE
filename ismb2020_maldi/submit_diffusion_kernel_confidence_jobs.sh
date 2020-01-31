#!/bin/sh

SEED=(58925 15250 97412 17965 44873)
MEMORY=8192
TIME=23:59

for S in "${SEED[@]}"; do
  OUTPUT="Calibration_saureus_seed${S}_GP_diffusion"
  bsub -N -W $TIME -o "${OUTPUT}_%J.json" -R "rusage[mem=${MEMORY}]" "poetry run python diffusion_kernel_confidence.py --sigma 16.22 --seed $S"
done
