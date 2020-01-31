#!/usr/bin/env python3
#
# Trains a diffusion kernel Gaussian Process classifier and reports the
# results on all tasks.

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
from sklearn.metrics import accuracy_score

from imblearn.over_sampling import RandomOverSampler

from joblib import parallel_backend

import numpy as np
import json_tricks as jt

import argparse
import os
import warnings


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--species', type=str, required=True)
    parser.add_argument('-a', '--antibiotic', type=str, required=True)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('-S', '--seed', type=int, required=False,
            default=2020)
    parser.add_argument('-p', '--peaks', type=int, required=False,
            default=None)
    parser.add_argument('-n', '--normalize', action='store_true')
    parser.add_argument('--suffix', default='')

    args = parser.parse_args()

    species_to_dataset = {
        'ecoli': EcoliAntibioticResistanceDataset,
        'kpneu': KpneuAntibioticResistanceDataset,
        'saureus': SaureusAntibioticResistanceDataset
    }

    dataset = species_to_dataset[args.species](
                test_size=0.20,
                antibiotic=args.antibiotic,
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

    # Normalise on demand. This is an *external* flag because by
    # default, we should have no expectations about its efficacy
    # in practice.
    if args.normalize:
        tic = TotalIonCurrentNormalizer(method='sum')
        X_train = tic.fit_transform(X_train)
        X_test = tic.transform(X_test)

    # Sparsify the data by restricting everything to the peaks only.
    st = SubsetPeaksTransformer(n_peaks=args.peaks)

    X_train = st.fit_transform(X_train)
    X_test = st.transform(X_test)

    # Perform scale normalisation for the MQ data set, which is
    # indicated by a suffix, or whenever the client specified a
    # normalisation parameter manually. This ensures that every
    # spectrum can be fitted by the kernel.
    if args.normalize or len(args.suffix) > 0:
        sn = ScaleNormalizer()
        X_train = sn.fit_transform(X_train)
        X_test = sn.transform(X_test)

    # Static information about the data set; will be extended later on
    # with information about the training itself.
    data = {
        'seed': args.seed,
        'species': args.species,
        'antibiotic': args.antibiotic,
        'n_peaks': args.peaks,
        'spectra_path': os.getenv('ANTIBIOTICS_SPECTRA_PATH'),
        'endpoint_path': os.getenv('ANTIBIOTICS_ENDPOINT_PATH'),
        'normalize': args.normalize,
    }

    kernel = DiffusionKernel(sigma=1)
    clf = GaussianProcessClassifier(kernel=kernel)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

        # Let's do the fitting in parallel, but the prediction can be done
        # without additional threading.
        with parallel_backend(backend='loky'):
            clf.fit(X_train, y_train)

        data['kernel'] = repr(clf.kernel_)
        data['log_marginal_likelihood'] = clf.log_marginal_likelihood_value_

    y_pred = clf.predict_proba(X_test)
    average_precision = average_precision_score(y_test, y_pred[:, 1])

    data['average_precision'] = 100 * average_precision

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    data['accuracy'] = 100 * accuracy

    if args.output is not None:
        with open(args.output, 'w') as f:
            jt.dump(data, f, indent=4)
    else:
        print(jt.dumps(data, indent=4))
