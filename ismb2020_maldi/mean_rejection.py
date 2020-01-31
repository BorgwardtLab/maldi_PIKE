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

    mean_rejected_in_sample = df[['rejected_in_sample', 'rejected_in_sample_1', 'rejected_in_sample_2', 'rejected_in_sample_3',
        'rejected_in_sample_4']].mean(axis=1)

    std_rejected_in_sample = df[['rejected_in_sample', 'rejected_in_sample_1', 'rejected_in_sample_2', 'rejected_in_sample_3',
        'rejected_in_sample_4']].std(axis=1)

    mean_rejected_out_of_sample = df[['rejected_out_of_sample',
        'rejected_out_of_sample_1',
        'rejected_out_of_sample_2',
        'rejected_out_of_sample_3',
        'rejected_out_of_sample_4']].mean(axis=1)

    std_rejected_out_of_sample = df[['rejected_out_of_sample',
        'rejected_out_of_sample_1',
        'rejected_out_of_sample_2',
        'rejected_out_of_sample_3',
        'rejected_out_of_sample_4']].std(axis=1)

    df = pd.DataFrame({'mean_rejected_in_sample': mean_rejected_in_sample, 'std_rejected_in_sample': std_rejected_in_sample,
        'mean_rejected_out_of_sample': mean_rejected_out_of_sample,
        'std_rejected_out_of_sample': std_rejected_out_of_sample})

    df.to_csv(sys.stdout)
