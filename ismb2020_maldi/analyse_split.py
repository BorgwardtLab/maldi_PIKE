#!/usr/bin/env python
#
# Analyses a given split of a data set and prints some summary
# statistics. This is mean for debugging purposes only.

from ismb2020_maldi.datasets import AntibioticResistanceDataset
from ismb2020_maldi.datasets import EcoliAntibioticResistanceDataset
from ismb2020_maldi.datasets import KpneuAntibioticResistanceDataset
from ismb2020_maldi.datasets import SaureusAntibioticResistanceDataset

from maldi_learn.preprocessing import SubsetPeaksTransformer

from joblib import parallel_backend

import numpy as np

import argparse
import os
import warnings


def get_mean_and_std(X, y, l):

    intensities = []

    for spectrum, label in zip(X, y):
        if label == l:
            intensities.extend(spectrum[:, 1])

    return np.mean(intensities), np.std(intensities)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--species', type=str, required=True)
    parser.add_argument('-a', '--antibiotic', type=str, required=True)
    parser.add_argument('-S', '--seed', type=int, required=False,
            default=2020)
    parser.add_argument('-p', '--peaks', type=int, required=False,
            default=100)

    args = parser.parse_args()

    species_to_dataset = {
        'ecoli': EcoliAntibioticResistanceDataset,
        'kpneu': KpneuAntibioticResistanceDataset,
        'saureus': SaureusAntibioticResistanceDataset
    }

    dataset = species_to_dataset[args.species](
                test_size=0.20,
                antibiotic=args.antibiotic,
                random_seed=args.seed
            )

    X_train, y_train = dataset.training_data
    X_test, y_test = dataset.testing_data

    st = SubsetPeaksTransformer(n_peaks=args.peaks)

    X_train = st.fit_transform(X_train)
    X_test = st.transform(X_test)

    print(f'Seed: {args.seed}')
    print(f'Species: {args.species}')
    print(f'Antibiotic: {args.antibiotic}')
    print(f'Number of peaks: {args.peaks}')

    SPECTRA_PATH = os.getenv('ANTIBIOTICS_SPECTRA_PATH')
    ENDPOINT_PATH = os.getenv('ANTIBIOTICS_ENDPOINT_PATH')

    print(f'SPECTRA_PATH = {SPECTRA_PATH}')
    print(f'ENDPOINT_PATH = {ENDPOINT_PATH}')

    mu_train_0, std_train_0 = get_mean_and_std(X_train, y_train, 0)
    mu_train_1, std_train_1 = get_mean_and_std(X_train, y_train, 1)
    mu_test_0, std_test_0 = get_mean_and_std(X_test, y_test, 0)
    mu_test_1, std_test_1 = get_mean_and_std(X_test, y_test, 1)

    print('y == 0 (train):', mu_train_0, std_train_0)
    print('y == 1 (train):', mu_train_1, std_train_1)
    print('y == 0 (test):', mu_test_0, std_test_0)
    print('y == 1 (test):', mu_test_1, std_test_1)

    print(abs(mu_train_0 - mu_test_0), abs(std_test_0 - std_test_1))
