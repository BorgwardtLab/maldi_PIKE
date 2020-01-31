#!/usr/bin/env python3
#
# Basic visualisation script for baseline classifier.

from ismb2020_maldi.datasets import AntibioticResistanceDataset
from ismb2020_maldi.datasets import EcoliAntibioticResistanceDataset
from ismb2020_maldi.datasets import KpneuAntibioticResistanceDataset
from ismb2020_maldi.datasets import SaureusAntibioticResistanceDataset

from maldi_learn.vectorization import BinningVectorizer

from sklearn.manifold import TSNE

from joblib import parallel_backend

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import json_tricks as jt

import argparse
import os
import warnings


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--species', type=str, required=True)
    parser.add_argument('-a', '--antibiotic', type=str, required=True)
    parser.add_argument('-S', '--seed', type=int, required=False,
            default=2020)
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

    bv = BinningVectorizer(900, min_bin=2000, max_bin=20000)
    X_train = bv.fit_transform(X_train)
    X_test = bv.transform(X_test)

    X = np.concatenate((X_train, X_test), axis=0)

    # Static information about the data set; will be extended later on
    # with information about the training itself.
    data = {
        'seed': args.seed,
        'species': args.species,
        'antibiotic': args.antibiotic,
        'spectra_path': os.getenv('ANTIBIOTICS_SPECTRA_PATH'),
        'endpoint_path': os.getenv('ANTIBIOTICS_ENDPOINT_PATH'),
    }

    tsne = TSNE(n_components=2)
    tsne.fit(X)

    Z = tsne.fit_transform(X)
    Z_train = Z[:len(X_train)]
    Z_test = Z[len(X_train):]

    fig, axes = plt.subplots(ncols=2)

    sns.scatterplot(x=Z_train[:, 0], y=Z_train[:, 1], hue=y_train,
            ax=axes[0])
    sns.scatterplot(x=Z_test[:, 0], y=Z_test[:, 1], hue=y_test,
            ax=axes[1])

    plt.show()

    if args.output is not None:
        with open(args.output, 'w') as f:
            jt.dump(data, f, indent=4)
    else:
        print(jt.dumps(data, indent=4))
