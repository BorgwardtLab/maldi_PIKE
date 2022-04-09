#!/usr/bin/env python3
#
# Trains a baseline logistic regression classifier and reports the
# results on all tasks. Uses pre-processed spectra, does incorporate
# clustering information for the train-test split and does not do
# any peak calling in the prediction task.

from ismb2020_maldi.datasets import ClusterAntibioticResistanceDataset

from maldi_learn.vectorization import BinningVectorizer

from imblearn.over_sampling import RandomOverSampler

from sklearn.exceptions import FitFailedWarning
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
    parser.add_argument('-c', '--cluster_based_split', type=bool, default=False, required=True)
    parser.add_argument('-f', '--cluster_feature_selection', type=bool, default=False, required=False)
    parser.add_argument('-N', '--n_cluster', type=int, required=False, default=None)
    parser.add_argument('-l', '--cluster_linkage', type=str, required=False, default='ward')
    parser.add_argument('-S', '--seed', type=int, required=False, default=2020)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('--suffix', type=str, default='')

    args = parser.parse_args()

    dataset = ClusterAntibioticResistanceDataset(
                species=args.species,
                antibiotic=args.antibiotic,
                test_size=0.20,
                cluster_based_split=args.cluster_based_split,
                cluster_feature_selection=args.cluster_feature_selection,
                bv=BinningVectorizer(n_bins=6000,
                                     min_bin=2000,
                                     max_bin=20000),
                n_cluster=args.n_cluster,
                cluster_linkage=args.cluster_linkage,
                random_seed=args.seed,
                suffix=args.suffix
    )

    X_train, y_train = dataset.X_train, dataset.y_train
    X_test, y_test = dataset.X_test, dataset.y_test

    # Static information about the data set; will be extended later on
    # with information about the training itself.
    data = {
        'seed': args.seed,
        'species': args.species,
        'antibiotic': args.antibiotic,
        'driams_root': os.getenv('DRIAMS_ROOT_PATH'),
        'cluster_data_dir': os.getenv('CLUSTER_DATA_DIR'),
        'classification_data_dir': os.getenv('CLASSIFICATION_DATA_DIR'),
    }

    data['linkage'] = dataset.linkage
    data['n_cluster'] = dataset.n_cluster

    if getattr(dataset, 'fs', None):
        data['feature_selection'] = True

    if getattr(dataset, 'davies_bouldin_score', None):
        data['davies_bouldin_score'] = dataset.davies_bouldin_score
        data['silhouette_score'] = dataset.silhouette_score
        data['calinski_harabasz_score'] = dataset.calinski_harabasz_score

    # Perform random oversampling in order to ensure class balance. This
    # is strictly speaking not required but we do it for the GP as well,
    # so in the interest of comparability, we have to do it here.

    ros = RandomOverSampler(random_state=args.seed)

    X_indices = np.asarray(
        [i for i in range(0, len(X_train))]).reshape(-1, 1)

    X_indices, y_train = ros.fit_sample(X_indices, y_train)
    X_train = np.asarray(X_train, dtype=object)
    X_train = np.take(X_train, X_indices.ravel())

    param_grid = {
        'bv__n_bins': [300, 600, 1800, 3600],
        'lr__penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'lr__C': 10. ** np.arange(-4, 5),  # 10^{-4}..10^{4}
    }

    data['param_grid'] = param_grid

    # Define pipeline and cross-validation setup

    pipeline = Pipeline(
        [
            ('bv', BinningVectorizer(
                    n_bins=0,
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
        warnings.filterwarnings('ignore', category=FitFailedWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

        # Let's do the fitting in parallel, but the prediction can be done
        # without additional threading.
        with parallel_backend('threading'):
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
