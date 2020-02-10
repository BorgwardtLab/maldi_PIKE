#!/usr/bin/env python3
#
# Basic visualisation script for the feature map of our kernel, subject
# to a certain smoothing parameter.

from maldi_learn.data import MaldiTofSpectrum

from maldi_learn.preprocessing import ScaleNormalizer

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

import argparse


def feature_map(spectrum, x, sigma=1.0):
    positions = spectrum.mass_to_charge_ratios
    peaks = spectrum.intensities

    f = np.multiply(peaks, np.exp(-(x - positions)**2 / (4 * sigma)))
    f = 1 / (2 * np.sqrt(np.pi * sigma)) * np.sum(f)

    return f


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', type=str)

    args = parser.parse_args()

    spectrum = MaldiTofSpectrum(
        pd.read_csv(args.FILE, sep=' ', comment='#').values
    )

    sn = ScaleNormalizer()
    spectrum = sn.fit_transform([spectrum])[0]

    spectrum = spectrum[spectrum.mass_to_charge_ratios < 2500]
    x_min = np.min(spectrum.mass_to_charge_ratios)
    x_max = np.max(spectrum.mass_to_charge_ratios)

    fig, ax = plt.subplots(4, 1, sharex=True)

    ax[0].stem(spectrum.mass_to_charge_ratios, spectrum.intensities,
              linefmt='k-', basefmt='black', markerfmt='None',
              use_line_collection=True)

    for axis in ax:
        axis.set_ylim(0, 6)

        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)

        axis.set_yticks([0, 2, 4, 6])

    for axis, sigma in zip(ax[1:], [1, 10, 100]):

        X = np.linspace(x_min, x_max, 300)
        Y = [feature_map(spectrum, x, sigma) for x in X]

        axis.plot(X, Y)


    plt.show()
