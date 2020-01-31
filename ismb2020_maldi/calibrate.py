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

    thresholds = np.linspace(0.5, 1.0, 1000)

    test_proba = data['in_sample_test_proba']
    test_proba_max = np.amax(test_proba, axis=1)

    output_rejection_ratio_curve = \
        f'Calibration_{in_sample_species}_{antibiotic}_{seed}.csv'

    with open(output_rejection_ratio_curve, 'w') as f:

        print('threshold,accuracy,auprc,n_pos_samples', file=f)

        for threshold in thresholds:

            # Get the indices that we want to *keep*, i.e. those test
            # samples whose maximum probability exceeds the threshold
            indices = test_proba_max > threshold

            # Subset the predictions and the labels according to these
            # indices and calculate an AUPRC.
            y_true = y_test[indices]
            y_pred_proba = test_proba[indices][:, 1]

            # Predict the positive class if the prediction threshold is
            # larger than the one we use for this iteration.
            y_pred = np.zeros_like(y_pred_proba)
            y_pred[y_pred_proba > threshold] = 1.0

            y_true_unique = set(y_true.values)

            if len(y_true_unique) != 2:
                break

            average_precision = average_precision_score(y_true, y_pred_proba)
            accuracy = accuracy_score(y_true, y_pred)

            print(f'{threshold},{accuracy},{average_precision},{sum(y_true == 1)}', file=f)

    oos_species = data['out_of_sample_species']

    for species in oos_species:

        output_rejection_plot = \
            f'Rejection_ratio_{in_sample_species}_{antibiotic}_{species}_{seed}.csv'

        oos_proba = data['out_of_sample_' + species + '_proba']
        oos_proba_max = np.amax(oos_proba, axis=1)

        with open(output_rejection_plot, 'w') as f:

            print('threshold,rejected_in_sample,rejected_out_of_sample',
                    file=f)

            for threshold in thresholds:
                rejected_test = \
                    sum(test_proba_max <= threshold) / len(test_proba_max)

                rejected_oos = \
                    sum(oos_proba_max <= threshold) / len(oos_proba_max)

                print(f'{threshold},{rejected_test},{rejected_oos}',
                        file=f)


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
