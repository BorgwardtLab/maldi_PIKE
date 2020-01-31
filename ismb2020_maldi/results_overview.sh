#!/bin/sh

# example usage:
# ANTIBIOTICS=Ceftriaxon SPECIES=ecoli ./results_overview.sh

if [ -z ${ANTIBIOTICS+x} ]; then
  ANTIBIOTICS=(Amoxicillin-Clavulansaeure Ciprofloxacin Ceftriaxon)
fi

if [ -z ${SPECIES+x} ]; then
  SPECIES=(ecoli saureus kpneu)
fi

for A in "${ANTIBIOTICS[@]}"; do
  for S in "${SPECIES[@]}"; do
    echo
    echo ---- ${A} -- ${S} ----
    echo - baseline on raw - 
    cat /cluster/work/borgw/ismb2020_maldi/results/raw/${A}_${S}_*.out | grep Average
    echo - baseline on preprocessed - 
    cat /cluster/work/borgw/ismb2020_maldi/results/preprocessed/${A}_${S}_*.out | grep Average
    echo - GP random oversampling 200 peaks - 
    cat /cluster/work/borgw/ismb2020_maldi/results/diffusion_ros/GP_diffusion_${A}_${S}_*_200_*.out | grep Average

  done
done
