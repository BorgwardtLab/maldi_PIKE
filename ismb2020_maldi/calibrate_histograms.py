#!/usr/bin/env python
#
# Calculates calibration scores of the model by calculating changes in
# AUPRC depending on the threshold.

from ismb2020_maldi.datasets import AntibioticResistanceDataset
from ismb2020_maldi.datasets import EcoliAntibioticResistanceDataset
from ismb2020_maldi.datasets import KpneuAntibioticResistanceDataset
from ismb2020_maldi.datasets import SaureusAntibioticResistanceDataset

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score

import argparse

import numpy as np
import json_tricks as jt

from tqdm import tqdm


def process(data, in_sample_species, antibiotic, seed):

    species_to_dataset = {
        'ecoli': EcoliAntibioticResistanceDataset,
        'kpneu': KpneuAntibioticResistanceDataset,
        'saureus': SaureusAntibioticResistanceDataset
    }

    dataset = species_to_dataset[in_sample_species](
        test_size=0.20,
        antibiotic=antibiotic,
        random_seed=seed
    )

    _, y_test = dataset.testing_data

    test_proba = data['in_sample_test_proba']
    test_proba_max = np.amax(test_proba, axis=1)

    output_in_sample_proba = \
        f'In_sample_proba_{in_sample_species}_{antibiotic}_{seed}.csv'

    np.savetxt(output_in_sample_proba, test_proba_max, fmt='%.2f')

    oos_species = data['out_of_sample_species']

    for species in oos_species:

        oos_proba = data['out_of_sample_' + species + '_proba']
        oos_proba_max = np.amax(oos_proba, axis=1)

        output_out_of_sample_proba = \
            f'Out_of_sample_proba_{species}_{antibiotic}_{seed}.csv'

        np.savetxt(output_out_of_sample_proba, oos_proba_max, fmt='%.2f')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('FILES', nargs='+', type=str)

    args = parser.parse_args()

    for filename in tqdm(args.FILES, desc='Loading'):
        with open(filename) as f:
            data = jt.load(f)
            species = data['in_sample_species']
            antibiotic = data['in_sample_antibiotic']
            seed = data['seed']

            process(data, species, antibiotic, seed)
