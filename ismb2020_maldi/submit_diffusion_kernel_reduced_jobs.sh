#!/bin/bash
#
# Submission script for the _reduced_ set of diffusion kernel jobs, i.e.
# we do _not_ change the number of peaks and always use normalisation. A
# scenario like this closely matches that of MQ data.

SEED=(58925 15250 97412 17965 44873)
PEAKS=200
MEMORY=8192
TIME=23:59

# s. aureus

for A in Amoxicillin-Clavulansaeure Ciprofloxacin Penicillin; do
  for S in "${SEED[@]}"; do
    OUTPUT="${A}_saureus_seed${S}_peaks${P}_GP_diffusion"
    bsub -N -W $TIME -o "${OUTPUT}_normalized_%J.json" -R "rusage[mem=${MEMORY}]" "poetry run python diffusion_kernel.py --species saureus --antibiotic $A --seed $S --peaks ${PEAKS} --normalize"
  done
done

# e. coli

for A in Amoxicillin-Clavulansaeure Ciprofloxacin Ceftriaxon; do
  for S in "${SEED[@]}"; do
    OUTPUT="${A}_ecoli_seed${S}_peaks${P}_GP_diffusion"
    bsub -N -W $TIME -o "${OUTPUT}_normalized_%J.json" -R "rusage[mem=${MEMORY}]" "poetry run python diffusion_kernel.py --species ecoli --antibiotic $A --seed $S --peaks ${PEAKS} --normalize"
  done
done

# k. pneu

for A in Ciprofloxacin Ceftriaxon Piperacillin-Tazobactam; do
  for S in "${SEED[@]}"; do
    OUTPUT="${A}_kpneu_seed${S}_peaks${P}_GP_diffusion"
    bsub -N -W $TIME -o "${OUTPUT}_normalized_%J.json" -R "rusage[mem=${MEMORY}]" "poetry run python diffusion_kernel.py --species kpneu --antibiotic $A --seed $S --peaks ${PEAKS} --normalize"
  done
done
