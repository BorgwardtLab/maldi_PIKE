#!/usr/bin/env
#
# Auxiliary script for extracting baseline parameters from a set of runs
# and reporting their mean. This is useful to run a calibration with
# a pre-selected model.

import argparse
import re

import json_tricks as jt
import numpy as np

from tqdm import tqdm


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

        seed = data_raw['seed']
        best_parameters = data_raw['best_parameters']
        print(f'Seed {seed}: {best_parameters}')
