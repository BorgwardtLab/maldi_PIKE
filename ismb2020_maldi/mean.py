#!/usr/bin/env python
#
# Calculates the mean of a set of CSVs. The CSV files are assumed to
# contain the same ranges.

import argparse
import sys

import pandas as pd


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('FILES', nargs='+', type=str)

    args = parser.parse_args()

    data = []

    for filename in args.FILES:
        df = pd.read_csv(filename, header=0, index_col=0)
        data.append(df)

    df = data[0]
    columns = df.columns

    for index, right in enumerate(data[1:]):
        df = pd.merge(df, right,
                suffixes=('', '_' + str(index + 1)),
                how='outer', on=['threshold']
        )

    df = df.fillna(1.0)

    mean_auprc = df[['auprc', 'auprc_1', 'auprc_2', 'auprc_3',
        'auprc_4']].mean(axis=1)

    std_auprc = df[['auprc', 'auprc_1', 'auprc_2', 'auprc_3',
        'auprc_4']].std(axis=1)


    mean_accuracy = df[['accuracy', 'accuracy_1', 'accuracy_2', 'accuracy_3',
        'accuracy_4']].mean(axis=1)

    std_accuracy = df[['accuracy', 'accuracy_1', 'accuracy_2', 'accuracy_3',
        'accuracy_4']].std(axis=1)

    mean_n_pos_samples = df[['n_pos_samples',
        'n_pos_samples_1',
        'n_pos_samples_2',
        'n_pos_samples_3',
        'n_pos_samples_4']].mean(axis=1)

    df = pd.DataFrame({'mean_auprc': mean_auprc, 'std_auprc': std_auprc,
        'mean_n_pos_samples': mean_n_pos_samples, 'mean_accuracy':
        mean_accuracy, 'std_accuracy': std_accuracy})

    df.to_csv(sys.stdout)
