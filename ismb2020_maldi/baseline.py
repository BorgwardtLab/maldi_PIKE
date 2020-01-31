#!/usr/bin/env python3
#
# Trains a baseline logistic regression classifier and reports the
# results on all tasks.

from ismb2020_maldi.datasets import AntibioticResistanceDataset
from ismb2020_maldi.datasets import EcoliAntibioticResistanceDataset
from ismb2020_maldi.datasets import KpneuAntibioticResistanceDataset
from ismb2020_maldi.datasets import SaureusAntibioticResistanceDataset

from maldi_learn.preprocessing import TotalIonCurrentNormalizer
from maldi_learn.preprocessing import SubsetPeaksTransformer
from maldi_learn.vectorization import BinningVectorizer

from imblearn.over_sampling import RandomOverSampler

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
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
    parser.add_argument('-s', '--species', type=str, required=True)
    parser.add_argument('-a', '--antibiotic', type=str, required=True)
    parser.add_argument('-S', '--seed', type=int, required=False, default=2020)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('-n', '--normalize', action='store_true')

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


    # Static information about the data set; will be extended later on
    # with information about the training itself.
    data = {
        'seed': args.seed,
        'species': args.species,
        'antibiotic': args.antibiotic,
        'spectra_path': os.getenv('ANTIBIOTICS_SPECTRA_PATH'),
        'endpoint_path': os.getenv('ANTIBIOTICS_ENDPOINT_PATH'),
        'normalize': args.normalize,
    }

    param_grid = [
        {
            'pt__n_peaks': [50, 100, 200, 500, None],
            'bv__n_bins': [75, 150, 300, 600, 1800, 3600],
            'lr__penalty': ['l1', 'l2'],
            'lr__C': 10. ** np.arange(-4, 5),  # 10^{-4}..10^{4}
        },
        {
            'pt__n_peaks': [50, 100, 200, 500, None],
            'bv__n_bins': [75, 150, 300, 600, 1800, 3600],
            'lr__penalty': ['none'],
        }
    ]

    data['param_grid'] = param_grid

    # Define pipeline and cross-validation setup

    pipeline = Pipeline(
        [
            ('pt', SubsetPeaksTransformer(n_peaks=0)),
            ('bv', BinningVectorizer(
                    n_bins=3600,
                    min_bin=2000,
                    max_bin=20000)),
            ('std', StandardScaler()),
            ('lr', LogisticRegression(
                        class_weight='balanced',
                        solver='saga'  # supports L_1 and L_2 penalties
                   )
            )
        ],
        memory=os.getenv('TMPDIR', default=None),
    )

    grid_search = GridSearchCV(
                    pipeline,
                    param_grid=param_grid,
                    scoring='average_precision',
                    cv=StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=42),
                    n_jobs=-1,
                )

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

        # Let's do the fitting in parallel, but the prediction can be done
        # without additional threading.
        with parallel_backend('loky', n_jobs=-1):
            grid_search.fit(X_train, y_train)

        data['best_parameters'] = grid_search.best_params_

    # AUPRC

    y_pred = grid_search.predict_proba(X_test)
    average_precision = average_precision_score(y_test, y_pred[:, 1])

    data['average_precision'] = 100 * average_precision

    # Accuracy

    y_pred = grid_search.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    data['accuracy'] = 100 * accuracy

    if args.output is not None:
        with open(args.output, 'w') as f:
            jt.dump(data, f, indent=4)
    else:
        print(jt.dumps(data, indent=4))
