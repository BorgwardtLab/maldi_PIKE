#!/bin/bash

SEED=(58925 15250 97412 17965 44873)
MEMORY=8192
TIME=23:59

# s. aureus

for A in Amoxicillin-Clavulansaeure Ciprofloxacin Penicillin; do
  for s in "${SEED[@]}"; do
    OUTPUT="${A}_saureus_seed${s}_normalized.json"
    nice poetry run python baseline.py --species saureus --antibiotic $A --seed $s --normalize --output ${OUTPUT} &
  done
done

# e. coli

for A in Amoxicillin-Clavulansaeure Ciprofloxacin Ceftriaxon; do
  for s in "${SEED[@]}"; do
    OUTPUT="${A}_ecoli_seed${s}_normalized.json"
    nice poetry run python baseline.py --species ecoli --antibiotic $A --seed $s --normalize --output ${OUTPUT} &
  done
done

# k. pneu

for A in Ciprofloxacin Ceftriaxon Piperacillin-Tazobactam; do
  for s in "${SEED[@]}"; do
    OUTPUT="${A}_kpneu_seed${s}_normalized.json"
    nice poetry run python baseline.py --species kpneu --antibiotic $A --seed $s --normalize --output ${OUTPUT} &
  done
done
