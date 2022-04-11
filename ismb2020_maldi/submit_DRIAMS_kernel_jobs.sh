#!/bin/bash
#
# Submission script for the _reduced_ set of diffusion kernel jobs, i.e.
# we do _not_ change the number of peaks and always use normalisation. A
# scenario like this closely matches that of MQ data.

SEED=(58925 15250 97412 17965 44873)
MEMORY=4096
TIME=500:00
NCLUST=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
c=1
LINKAGE=(ward average weighted single complete)
RESULTS_DIR="./results/results_DRIAMS/kernel/"

# s. aureus - Staphylococcus aureus

for A in Penicillin Ciprofloxacin "Amoxicillin-Clavulanic acid"; do
  for S in "${SEED[@]}"; do
    for N in "${NCLUST[@]}"; do
      for l in "${LINKAGE[@]}"; do
        OUTPUT="${RESULTS_DIR}${A}_saureus_seed${S}_GP_normalized_DRIAMS-A_${l}_CLUSTERS_${N}.json"
        if [ ! -f "${OUTPUT}" ]; then
          bsub -n 4 -W $TIME -R "rusage[mem=${MEMORY}]" "poetry run python diffusion_kernel_clustering.py -s 'Staphylococcus aureus' -a '$A' -S $S -N $N -c $c -l $l -f 1 --normalize --output '${OUTPUT}'"
        else
          echo "Experiment has already been conducted"
        fi
      done
    done
  done
done

# e. coli - Escherichia coli

for A in Ciprofloxacin Ceftriaxone "Amoxicillin-Clavulanic acid"; do
  for S in "${SEED[@]}"; do
    for N in "${NCLUST[@]}"; do
      for l in "${LINKAGE[@]}"; do
        OUTPUT="${RESULTS_DIR}${A}_ecoli_seed${S}_GP_normalized_DRIAMS-A_${l}_CLUSTERS_${N}.json"
        if [ ! -f "${OUTPUT}" ]; then
          bsub -n 4 -W $TIME -R "rusage[mem=${MEMORY}]" "poetry run python diffusion_kernel_clustering.py -s 'Escherichia coli' -a '$A' -S $S -N $N -c $c -l $l -f 1 --normalize --output '${OUTPUT}'"
        else
          echo "Experiment has already been conducted"
        fi
      done
    done
  done
done

# k. pneu - Klebsiella pneumoniae

for A in Ciprofloxacin Ceftriaxone "Piperacillin-Tazobactam"; do
  for S in "${SEED[@]}"; do
    for N in "${NCLUST[@]}"; do
      for l in "${LINKAGE[@]}"; do
        OUTPUT="${RESULTS_DIR}${A}_kpneu_seed${S}_GP_normalized_DRIAMS-A_${l}_CLUSTERS_${N}.json"
        if [ ! -f "${OUTPUT}" ]; then
          bsub -n 4 -W $TIME -R "rusage[mem=${MEMORY}]" "poetry run python diffusion_kernel_clustering.py -s 'Klebsiella pneumoniae' -a '$A' -S $S -N $N -c $c -l $l -f 1 --normalize --output '${OUTPUT}'"
        else
          echo "Experiment has already been conducted"
        fi
      done
    done
  done
done
