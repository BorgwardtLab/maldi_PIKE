"""Utility functions."""

import logging
import os

import numpy as np
import pandas as pd
from scipy.sparse import data
import seaborn as sns

from maldi_learn.driams import load_driams_dataset

from maldi_learn.filters import DRIAMSBooleanExpressionFilter

from maldi_learn.utilities import case_based_stratification
from maldi_learn.utilities import stratify_by_species_and_label

from sklearn.utils import shuffle

def case_cluster_based_stratification(
    y,
    class_cluster_labels,
    antibiotic,
    test_size=0.20,
    return_stratification=False,
    random_state=123
):
    """Stratify while taking patient case information and cluster labels into account."""
    # Ensuring proper cast to ensure that we can always perform mean
    # aggregation later on.
    y_copy = y.copy()
    y_copy[antibiotic] = y_copy[antibiotic].astype(float)
        
    # Add cluster labels to metadata
    y_copy['cluster'] = class_cluster_labels[:, 1]

    # Use mean class label and most common cluster label for each case
    unique_groups = y_copy.groupby('case_no').aggregate(
        {
            antibiotic: 'mean',
            'species': 'first',
            'cluster': lambda x: x.value_counts().index[0]
        }
    )
    unique_groups[antibiotic] = unique_groups[antibiotic].round()

    # Combine cluster and class labels into one, e.g. cluster 3, class 0 results in 30
    unique_groups[antibiotic] = unique_groups[antibiotic].astype(int)
    unique_groups[antibiotic] = (unique_groups['cluster'].astype(str) + 
        unique_groups[antibiotic].astype(str)).astype(int)

    y_copy = y_copy.reset_index(drop=True)

    # By default, we always use the returned stratification here, making
    # it possible to use it later on.
    train_index, test_index, train_labels, test_labels = \
        stratify_by_species_and_label(
            unique_groups,
            antibiotic=antibiotic,
            test_size=test_size,
            random_state=random_state,
            return_stratification=True,
        )

    train_index = unique_groups.iloc[train_index]
    test_index = unique_groups.iloc[test_index]

    # Make the case_no column, which has become an index, into
    # a column again.
    train_index.reset_index(inplace=True)
    test_index.reset_index(inplace=True)

    # Get original case numbers belonging to each unique group. We need
    # to *expand* these groups subsequently.
    train_id = train_index['case_no'].values
    test_id = test_index['case_no'].values

    # Create a column that contains the unique labels of the train and
    # test data points, respectively. There are multiple ways to solve
    # this but this one requires no additional data frame.

    case_to_label = {}

    for ids, labels in zip([train_id, test_id], [train_labels, test_labels]):
        case_to_label.update({
            id_: label.tolist() for id_, label in zip(ids, labels)
         })

    # Auxiliary function to assign a label to a row. Since we might not
    # have labels for all cases available, we have to return a fake one
    # instead. Such labels will never be used for train/test, though.
    def get_label(row):
        case_no = row['case_no']
        if case_no in case_to_label:
            return case_to_label[case_no]
        else:
            return [-1, -1]

    y_copy['unique_label'] = y_copy.apply(get_label, axis=1)

    # The queries serve to expand the data points again. Everything that
    # belongs to the same case number will now be either assigned to the
    # train part or the test portion.

    train_index = y_copy.query('case_no in @train_id').index
    train_labels = y_copy.query('case_no in @train_id').unique_label

    test_index = y_copy.query('case_no in @test_id').index
    test_labels = y_copy.query('case_no in @test_id').unique_label

    train_index, train_labels = shuffle(
        train_index, train_labels,
        random_state=random_state
    )

    test_index, test_labels = shuffle(
        test_index, test_labels,
        random_state=random_state
    )

    if return_stratification:
        return train_index, test_index, train_labels, test_labels
    else:
        return train_index, test_index


def load_data(
    root,
    site,
    years,
    species,
    antibiotic,
    spectra_type
):
    """Load data without additional splits"""
    extra_filters = []
    extra_filters.append(
        DRIAMSBooleanExpressionFilter('workstation != HospitalHygiene')
    )

    id_suffix = 'strat'

    dataset = load_driams_dataset(
        root,
        site,
        years=years,
        species=species,
        antibiotics=antibiotic,
        handle_missing_resistance_measurements='remove_if_all_missing',
        spectra_type=spectra_type,
        on_error='warn',
        id_suffix=id_suffix,
        extra_filters=extra_filters,
    )

    logging.info(f'Loaded data set for {species} and {antibiotic}')
    return dataset


def split_data(
    antibiotic,
    test_size,
    seed,
    dataset,
    metadata,
    class_cluster_labels
):
    """Split data set and return it in partitioned form."""

    # Use cluster labels in addition to case and class label only if requested
    if class_cluster_labels is None :
        train_index, test_index = case_based_stratification(
            metadata,
            antibiotic=antibiotic,
            test_size=test_size,
            random_state=seed
        )
    else:
        train_index, test_index = case_cluster_based_stratification(
            metadata,
            antibiotic=antibiotic,
            test_size=test_size,
            random_state=seed,
            class_cluster_labels=class_cluster_labels
        )

    # The maldi-learn methods used in this project expect a 2D array, not only the intensities
    X = np.asarray(dataset)

    # Use the column containing antibiotic information as the primary
    # label for the experiment. All other columns will be considered
    # metadata. The remainder of the script decides whether they are
    # being used or not.
    y = metadata[antibiotic].to_numpy(dtype='int')
    print("Class ratios ----------------------> ", np.unique(y, return_counts=True), "\n")
    print("Sample size -----------------------> ", y.size, "\n")
    #meta = metadata.drop(columns=antibiotic)

    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    #meta_train, meta_test = meta.iloc[train_index], meta.iloc[test_index]

    return X_train, y_train, X_test, y_test#, meta_train, meta_test
