#!/usr/bin/env python3
#
# Basic visualisation script for the feature map of our kernel, subject
# to a certain smoothing parameter.

from maldi_learn.data import MaldiTofSpectrum

from maldi_learn.preprocessing import ScaleNormalizer

from sklearn.decomposition import KernelPCA

from joblib import parallel_backend

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

    np.savetxt('Example_peaks.txt', spectrum, fmt='%.2f')

    for sigma in [1, 10, 100]:

        with open(f'Example_spectrum_smooth_sigma{sigma}.txt', 'w') as f:
            X = np.linspace(x_min, x_max, 300)
            Y = [feature_map(spectrum, x, sigma) for x in X]

            for x, y in zip(X, Y):
                print(x, y, file=f)

        plt.stem(spectrum.mass_to_charge_ratios, spectrum.intensities,
                linefmt='k-', basefmt='black')
        plt.plot(X, Y)

    plt.show()
