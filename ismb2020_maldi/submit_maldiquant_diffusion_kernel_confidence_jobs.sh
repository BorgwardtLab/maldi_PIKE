#!/bin/sh

SEED=(58925 15250 97412 17965 44873)
MEMORY=8192
TIME=23:59

export ANTIBIOTICS_SPECTRA_PATH=/cluster/work/borgw/ismb2020_maldi/spectra_MaldiQuant/
export ANTIBIOTICS_ENDPOINT_PATH=/cluster/work/borgw/ismb2020_maldi/

for S in "${SEED[@]}"; do
  OUTPUT="Calibration_saureus_seed${S}_MQ_GP_diffusion"
  bsub -N -W $TIME -o "${OUTPUT}_%J.json" -R "rusage[mem=${MEMORY}]" "poetry run python diffusion_kernel_confidence.py --sigma 4.18 --seed $S --suffix _peaks_warped"
done
