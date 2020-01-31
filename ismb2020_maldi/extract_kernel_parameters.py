#!/usr/bin/env
#
# Auxiliary script for extracting kernel parameters from a set of runs
# and reporting their mean. This is useful to run a calibration with a
# pre-selected model.

import argparse
import re

import json_tricks as jt
import numpy as np

from tqdm import tqdm


def extract_parameter(s, name='DiffusionKernel'):
    '''
    Extracts the kernel parameter from a string. The function attempts
    to extract a float value enclosed between brackets that correspond
    to a kernel name.

    Returns `np.nan` if no match could be found.
    '''

    pattern = f'{name}\((.+)\)'
    m = re.match(pattern, s)

    if m:
        return float(m.group(1))
    else:
        return np.nan


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT', nargs='+', type=str)
    args = parser.parse_args()

    parameters = []

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

        kernel = data_raw['kernel']
        parameter = extract_parameter(kernel)

        parameters.append(parameter)

    mu = np.mean(parameters)
    sigma = np.std(parameters)

    print('Extracted kernel parameters:', parameters)
    print(f'Mean kernel parameter: {mu:.2f} +- {sigma:.2f}')
