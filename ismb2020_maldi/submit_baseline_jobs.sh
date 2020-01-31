#!/bin/bash

SEED=(58925 15250 97412 17965 44873)
MEMORY=8192
TIME=23:59

# s. aureus

for A in Amoxicillin-Clavulansaeure Ciprofloxacin Penicillin; do
  for s in "${SEED[@]}"; do
    OUTPUT="${A}_saureus_seed${s}"
    bsub -N -W $TIME -o "${OUTPUT}_%J.json" -R "rusage[mem=${MEMORY}]" "poetry run python baseline.py --species saureus --antibiotic $A --seed $s"
    bsub -N -W $TIME -o "${OUTPUT}_normalized_%J.json" -R "rusage[mem=${MEMORY}]" "poetry run python baseline.py --species saureus --antibiotic $A --seed $s --normalize"
  done
done

# e. coli

for A in Amoxicillin-Clavulansaeure Ciprofloxacin Ceftriaxon; do
  for s in "${SEED[@]}"; do
    OUTPUT="${A}_ecoli_seed${s}"
    bsub -N -W $TIME -o "${OUTPUT}_%J.json" -R "rusage[mem=${MEMORY}]" "poetry run python baseline.py --species ecoli --antibiotic $A --seed $s"
    bsub -N -W $TIME -o "${OUTPUT}_normalized_%J.json" -R "rusage[mem=${MEMORY}]" "poetry run python baseline.py --species ecoli --antibiotic $A --seed $s --normalize"
  done
done

# k. pneu

for A in Ciprofloxacin Ceftriaxon Piperacillin-Tazobactam; do
  for s in "${SEED[@]}"; do
    OUTPUT="${A}_kpneu_seed${s}"
    bsub -N -W $TIME -o "${OUTPUT}_%J.json" -R "rusage[mem=${MEMORY}]" "poetry run python baseline.py --species kpneu --antibiotic $A --seed $s"
    bsub -N -W $TIME -o "${OUTPUT}_normalized_%J.json" -R "rusage[mem=${MEMORY}]" "poetry run python baseline.py --species kpneu --antibiotic $A --seed $s --normalize"
  done
done
