#!/usr/bin/env python3
#
# Trains a diffusion kernel Gaussian Process classifier and reports the
# results on all tasks.

from ismb2020_maldi.datasets import ClusterAntibioticResistanceDataset

from maldi_learn.kernels import DiffusionKernel
from maldi_learn.vectorization import BinningVectorizer
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
    parser.add_argument('-c', '--cluster_based_split', type=bool, default=False, required=True)
    parser.add_argument('-f', '--cluster_feature_selection', type=bool, default=False, required=False)
    parser.add_argument('-N', '--n_cluster', type=int, required=False, default=None)
    parser.add_argument('-l', '--cluster_linkage', type=str, required=False, default='ward')
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('-S', '--seed', type=int, required=False, default=2020)
    parser.add_argument('-n', '--normalize', action='store_true')
    parser.add_argument('--suffix', default='')

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
        'n_peaks': None,
        'driams_root': os.getenv('DRIAMS_ROOT_PATH'),
        'cluster_data_dir': os.getenv('CLUSTER_DATA_DIR'),
        'classification_data_dir': os.getenv('CLASSIFICATION_DATA_DIR'),
        'normalize': args.normalize
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

    # Normalise on demand. This is an *external* flag because by
    # default, we should have no expectations about its efficacy
    # in practice.
    if args.normalize:
        tic = TotalIonCurrentNormalizer(method='sum')
        X_train = tic.fit_transform(X_train)
        X_test = tic.transform(X_test)

    # Sparsify the data by restricting everything to the peaks only.
    st = SubsetPeaksTransformer(n_peaks=None)

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
        # Create output directory if it does not exist yet
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        # Save results
        with open(args.output, 'w') as f:
            jt.dump(data, f, indent=4)
    else:
        print(jt.dumps(data, indent=4))
