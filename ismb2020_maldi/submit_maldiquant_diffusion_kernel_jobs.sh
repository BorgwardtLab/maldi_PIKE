#!/bin/sh

SEED=(58925 15250 97412 17965 44873)
MEMORY=8192
TIME=23:59

export ANTIBIOTICS_SPECTRA_PATH=/cluster/work/borgw/ismb2020_maldi/spectra_MaldiQuant/
export ANTIBIOTICS_ENDPOINT_PATH=/cluster/work/borgw/ismb2020_maldi/

# s. aureus

for A in Amoxicillin-Clavulansaeure Ciprofloxacin Penicillin; do
  for S in "${SEED[@]}"; do
    OUTPUT="${A}_saureus_seed${S}_peaks${P}_MQ_GP_diffusion"
    bsub -N -W $TIME -o "${OUTPUT}_%J.json" -R "rusage[mem=${MEMORY}]" "poetry run python diffusion_kernel.py --species saureus --antibiotic $A --seed $S --suffix _peaks_warped"
  done
done

# e. coli

for A in Amoxicillin-Clavulansaeure Ciprofloxacin Ceftriaxon; do
  for S in "${SEED[@]}"; do
    OUTPUT="${A}_ecoli_seed${S}_peaks${P}_MQ_GP_diffusion"
    bsub -N -W $TIME -o "${OUTPUT}_%J.json" -R "rusage[mem=${MEMORY}]" "poetry run python diffusion_kernel.py --species ecoli --antibiotic $A --seed $S --suffix _peaks_warped"
  done
done

# k. pneu

for A in Ciprofloxacin Ceftriaxon Piperacillin-Tazobactam; do
  for S in "${SEED[@]}"; do
    OUTPUT="${A}_kpneu_seed${S}_peaks${P}_MQ_GP_diffusion"
    bsub -N -W $TIME -o "${OUTPUT}_%J.json" -R "rusage[mem=${MEMORY}]" "poetry run python diffusion_kernel.py --species kpneu --antibiotic $A --seed $S --suffix _peaks_warped"
  done
done
