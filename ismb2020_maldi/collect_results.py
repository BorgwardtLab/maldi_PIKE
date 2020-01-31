#!/usr/bin/env python
#
# Collection script for all results. Will create a table based on the
# species and the antibiotic and summarise the performance measures.

import argparse

import json_tricks as jt
import numpy as np
import pandas as pd

from tqdm import tqdm


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', nargs='+', type=str)

    # Following the convention of `sklearn` here instead of referring to
    # AUPRC or something like that.
    parser.add_argument(
            '-m', '--metric',
            default='average_precision',
            type=str
    )

    args = parser.parse_args()

    rows = []

    for filename in tqdm(args.INPUT, desc='Loading'):
        with open(filename) as f:
            # Ensures that we can parse normal JSON files
            pos = 0

            for line in f:

                # We found *probably* the beginning of the JSON file, so
                # we can start the parse process from here, having to do
                # a reset.
                if line.startswith('{'):
                    f.seek(pos)
                    break
                else:
                    pos += len(line)

            # Check whether file is empty for some reason. If so, we
            # skip it.
            line = f.readline()
            if line == '':
                continue

            # Not empty, so we need to reset the file pointer
            else:
                f.seek(pos)

            data_raw = jt.load(f)

        # Create one row in the table containing the relevant
        # information for now.
        row = {
            'species': data_raw['species'],
            'antibiotic': data_raw['antibiotic'],
            args.metric: data_raw[args.metric],
        }

        # Some magic for figuring out whether we are looking at
        # a baseline method or one of our own. Also creates the
        # name of the method.
        #
        # TODO: handle multiple kernels

        is_baseline = 'best_parameters' in data_raw
        method = 'baseline' if is_baseline else 'kernel'

        if not is_baseline:
            if data_raw['n_peaks'] is None:
                data_raw['n_peaks'] = 'all'

            method += '_' + str(data_raw['n_peaks'])

        if 'normalize' in data_raw:
            method += '_normalized' if data_raw['normalize'] else ''

        row['method'] = method
        rows.append(row)

    pd.options.display.max_rows = 999
    pd.options.display.float_format = '{:,.2f}'.format

    df = pd.DataFrame(rows)
    df = df.groupby(['species', 'antibiotic', 'method']).agg(
        {
            args.metric: [np.mean, np.std]
        }
    )

    print(df)
