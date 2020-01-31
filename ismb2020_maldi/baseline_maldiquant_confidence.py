#!/usr/bin/env python3
#
# Trains a baseline logistic regression classifier and reports the
# confidence scores on our pre-defined task.

from ismb2020_maldi.datasets import AntibioticResistanceDataset
from ismb2020_maldi.datasets import EcoliAntibioticResistanceDataset
from ismb2020_maldi.datasets import KpneuAntibioticResistanceDataset
from ismb2020_maldi.datasets import SaureusAntibioticResistanceDataset

from maldi_learn.vectorization import BinningVectorizer

from imblearn.over_sampling import RandomOverSampler

from sklearn.exceptions import FitFailedWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from joblib import parallel_backend

import numpy as np
import json_tricks as jt

import argparse
import os
import warnings


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--seed', type=int, required=False, default=2020)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('--suffix', type=str, default='')

    args = parser.parse_args()

    in_sample_species = 'saureus'
    in_sample_antibiotic = 'Amoxicillin-Clavulansaeure'

    # Antibiotics will not be used, but are specified because our data
    # set selection class demands it.
    out_of_sample_species = ['ecoli', 'kpneu']
    out_of_sample_antibiotics = ['Ciprofloxacin', 'Ciprofloxacin']

    species_to_dataset = {
        'ecoli': EcoliAntibioticResistanceDataset,
        'kpneu': KpneuAntibioticResistanceDataset,
        'saureus': SaureusAntibioticResistanceDataset
    }

    dataset = species_to_dataset[in_sample_species](
                test_size=0.20,
                antibiotic=in_sample_antibiotic,
                random_seed=args.seed,
                suffix=args.suffix
            )

    X_train, y_train = dataset.training_data
    X_test, y_test = dataset.testing_data

    # Perform random oversampling in order to ensure class balance. This
    # is strictly speaking not required but we do it for the GP as well,
    # so in the interest of comparability, we have to do it here.

    ros = RandomOverSampler(random_state=args.seed)

    X_indices = np.asarray(
        [i for i in range(0, len(X_train))]).reshape(-1, 1)

    X_indices, y_train = ros.fit_sample(X_indices, y_train)
    X_train = np.take(X_train, X_indices.ravel())

    # Parameters extracted from the respective runs of the baseline
    # classifier for this particular scenario.
    n_bins = 3600
    C = 0.01
    penalty = 'l2'

    # Static information about the data set; will be extended later on
    # with information about the training itself.
    data = {
        'seed': args.seed,
        'in_sample_antibiotic': in_sample_antibiotic,
        'in_sample_species': in_sample_species,
        'out_of_sample_species': out_of_sample_species,
        'out_of_sample_antibiotics': out_of_sample_antibiotics,
        'spectra_path': os.getenv('ANTIBIOTICS_SPECTRA_PATH'),
        'endpoint_path': os.getenv('ANTIBIOTICS_ENDPOINT_PATH'),
        'n_bins': n_bins,
        'C': C,
        'penalty': penalty,
    }

    pipeline = Pipeline(
        [
            ('bv', BinningVectorizer(
                    n_bins=n_bins,
                    min_bin=2000,
                    max_bin=20000)),
            ('std', StandardScaler()),
            ('lr', LogisticRegression(
                        class_weight='balanced',
                        C=C,
                        penalty=penalty,
                        solver='saga'  # supports L_1 and L_2 penalties
                   )
            )
        ],
        memory=os.getenv('TMPDIR', default=None),
    )

    # Makes subsequent operations easier to read
    clf = pipeline

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', category=FitFailedWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

        # Let's do the fitting in parallel, but the prediction can be done
        # without additional threading.
        with parallel_backend('loky'):
            clf.fit(X_train, y_train)

    # Get maximum probability for classifying a sample into *any* class,
    # based on the test data set.
    test_proba = clf.predict_proba(X_test)
    test_proba_max = np.amax(test_proba, axis=1)

    data['in_sample_test_proba'] = test_proba

    for species, antibiotic in zip(out_of_sample_species,
            out_of_sample_antibiotics):

        oos_dataset = species_to_dataset[species](
            test_size=0.20,
            antibiotic=antibiotic,
            random_seed=args.seed,
            suffix=args.suffix,
        )

        oos_test, _ = oos_dataset.testing_data

        oos_proba = clf.predict_proba(oos_test)
        oos_proba_max = np.amax(oos_proba, axis=1)

        data['out_of_sample_' + species + '_proba'] = oos_proba

    if args.output is not None:
        with open(args.output, 'w') as f:
            jt.dump(data, f, indent=4)
    else:
        print(jt.dumps(data, indent=4))
