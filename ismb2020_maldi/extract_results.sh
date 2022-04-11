#!/bin/sh

LINKAGE=(ward single average complete weighted)

# s. aureus

for A in "Amoxicillin-Clavulanic acid" Ciprofloxacin Penicillin; do
    for L in "${LINKAGE[@]}"; do
        #python extract_results_csv.py -m baseline -s 'Staphylococcus aureus' -a "$A" -l "$L" ./results/results_DRIAMS/baseline/"$A"*saureus*json
        python extract_results_csv.py -m kernel -s 'Staphylococcus aureus' -a "$A" -l "$L" ./results/results_DRIAMS/kernel/"$A"*saureus*json
    done
done

# e. coli

for A in "Amoxicillin-Clavulanic acid" Ciprofloxacin Ceftriaxone; do
    for L in "${LINKAGE[@]}"; do
        #python extract_results_csv.py -m baseline -s 'Escherichia coli' -a "$A" -l "$L" ./results/results_DRIAMS/baseline/"$A"*ecoli*json
        python extract_results_csv.py -m kernel -s 'Escherichia coli' -a "$A" -l "$L" ./results/results_DRIAMS/kernel/"$A"*ecoli*json
    done
done

# k. pneu

for A in Ciprofloxacin Ceftriaxone Piperacillin-Tazobactam; do
    for L in "${LINKAGE[@]}"; do
        #python extract_results_csv.py -m baseline -s 'Klebsiella pneumoniae' -a "$A" -l "$L" ./results/results_DRIAMS/baseline/"$A"*kpneu*json
        python extract_results_csv.py -m kernel -s 'Klebsiella pneumoniae' -a "$A" -l "$L" ./results/results_DRIAMS/kernel/"$A"*kpneu*json
    done
done
