#!/usr/bin/env python3
#
# Performs a confidence estimation experiment on a subset of the data
# sets in order to check whether we may reject samples from another
# distribution.

from ismb2020_maldi.datasets import AntibioticResistanceDataset
from ismb2020_maldi.datasets import EcoliAntibioticResistanceDataset
from ismb2020_maldi.datasets import KpneuAntibioticResistanceDataset
from ismb2020_maldi.datasets import SaureusAntibioticResistanceDataset

from maldi_learn.kernels import DiffusionKernel
from maldi_learn.preprocessing import TotalIonCurrentNormalizer
from maldi_learn.preprocessing import SubsetPeaksTransformer
from maldi_learn.preprocessing import ScaleNormalizer

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score

from imblearn.over_sampling import RandomOverSampler

from joblib import parallel_backend

import numpy as np
import json_tricks as jt

import argparse
import os
import warnings


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('-s', '--sigma', type=float, required=True)
    parser.add_argument('-S', '--seed', type=int, default=2020)
    parser.add_argument('-n', '--normalize', action='store_true')

    # By default, we assume that we want to use *all* the peaks because
    # we are comparing our model to a pre-processed pipeline.
    parser.add_argument('-p', '--peaks', type=int, required=False,
            default=None)
    parser.add_argument('--suffix', default='')

    args = parser.parse_args()

    in_sample_species = 'saureus'
    in_sample_antibiotic = 'Amoxicillin-Clavulansaeure'
    n_peaks = args.peaks

    # Antibiotics will not be used, but are specified because our data
    # set selection class demands it.
    out_of_sample_species = ['ecoli', 'kpneu']
    out_of_sample_antibiotics = ['Ciprofloxacin', 'Ciprofloxacin']

    args = parser.parse_args()

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

    # Only perform scale normalisation if a suffix has been set; this
    # should be made configurable.
    if len(args.suffix) > 0:
        sn = ScaleNormalizer()
        X_train = sn.fit_transform(X_train)
        X_test = sn.transform(X_test)

    # Perform random oversampling in order to ensure class balance. This
    # is strictly speaking not required but we do it for the GP as well,
    # so in the interest of comparability, we have to do it here.

    ros = RandomOverSampler(random_state=args.seed)

    X_indices = np.asarray(
        [i for i in range(0, len(X_train))]).reshape(-1, 1)

    X_indices, y_train = ros.fit_sample(X_indices, y_train)
    X_train = np.take(X_train, X_indices.ravel())

    # Normalise on demand. This is an *external* flag because by
    # default, we should have no expectations about its efficacy
    # in practice.
    if args.normalize:
        tic = TotalIonCurrentNormalizer()
        X_train = tic.fit_transform(X_train)
        X_test = tic.transform(X_test)

    # Sparsify the data by restricting everything to the peaks only.
    st = SubsetPeaksTransformer(n_peaks=n_peaks)

    X_train = st.fit_transform(X_train)
    X_test = st.transform(X_test)

    kernel = DiffusionKernel(sigma=args.sigma)
    clf = GaussianProcessClassifier(kernel=kernel, optimizer=None)

    # Static information about the data set; will be extended later on
    # with information about the training itself.
    data = {
        'seed': args.seed,
        'in_sample_antibiotic': in_sample_antibiotic,
        'in_sample_species': in_sample_species,
        'out_of_sample_species': out_of_sample_species,
        'out_of_sample_antibiotics': out_of_sample_antibiotics,
        'n_peaks': n_peaks,
        'spectra_path': os.getenv('ANTIBIOTICS_SPECTRA_PATH'),
        'endpoint_path': os.getenv('ANTIBIOTICS_ENDPOINT_PATH'),
        'sigma': args.sigma,
    }

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

        # Let's do the fitting in parallel, but the prediction can be done
        # without additional threading.
        with parallel_backend(backend='loky'):
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
            suffix=args.suffix
        )

        oos_test, _ = oos_dataset.testing_data

        # Only perform scale normalisation if a suffix has been set; this
        # should be made configurable.
        if len(args.suffix) > 0:
            oos_test = sn.transform(oos_test)

        if args.normalize:
            oos_test = tic.transform(oos_test)

        oos_test = st.transform(oos_test)

        oos_proba = clf.predict_proba(oos_test)
        oos_proba_max = np.amax(oos_proba, axis=1)

        data['out_of_sample_' + species + '_proba'] = oos_proba

    if args.output is not None:
        with open(args.output, 'w') as f:
            jt.dump(data, f, indent=4)
    else:
        print(jt.dumps(data, indent=4))
