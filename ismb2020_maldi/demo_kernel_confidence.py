'''
Demo file for kernel calculation.
'''

from ismb2020_maldi.datasets import KpneuAntibioticResistanceDataset
from ismb2020_maldi.datasets import SaureusAntibioticResistanceDataset
from ismb2020_maldi.datasets import EcoliAntibioticResistanceDataset

from maldi_learn.kernels import DiffusionKernel
from maldi_learn.preprocessing import TotalIonCurrentNormalizer

from maldi_learn.preprocessing import SubsetPeaksTransformer

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import average_precision_score
from sklearn.svm import SVC

from imblearn.over_sampling import RandomOverSampler

from joblib import parallel_backend

import matplotlib.pyplot as plt

import numpy as np
import sys

dataset = KpneuAntibioticResistanceDataset(antibiotic='Ceftriaxon',
        test_size=0.20)
X_train, y_train = dataset.training_data
X_test, y_test = dataset.testing_data

X_indices = np.asarray([i for i in range(0,
    len(X_train))]).reshape(-1, 1)

ros = RandomOverSampler(random_state=2020)

X_indices, y_train = ros.fit_sample(X_indices, y_train)

X_train_ = []

for index in X_indices.ravel():
    X_train_.append(X_train[index])

X_train = X_train_

tic = TotalIonCurrentNormalizer()
X_train = tic.fit_transform(X_train)
X_test = tic.transform(X_test)

st = SubsetPeaksTransformer(n_peaks=200)

X_train = st.fit_transform(X_train)
X_test = st.transform(X_test)

kernel = DiffusionKernel(sigma=10)

print('Finished pre-processing')

test_distribution = kernel(X_train, X_test).ravel()

clf = GaussianProcessClassifier(optimizer=None, kernel=kernel, n_jobs=-1)
clf.fit(X_train, y_train)

test_distribution = np.amax(clf.predict_proba(X_test), axis=1).ravel()

oos_dataset = SaureusAntibioticResistanceDataset(antibiotic='Penicillin',
        test_size=0.20)

oos_test, _ = oos_dataset.testing_data
oos_test = tic.transform(oos_test)
oos_test = st.transform(oos_test)

oos_distribution = np.amax(clf.predict_proba(oos_test), axis=1).ravel()

plt.hist(test_distribution, label='test', bins=np.linspace(0.50, 0.60, 100), alpha=0.5)
plt.hist(oos_distribution, label='oos saureus', bins=np.linspace(0.50, 0.60, 100), alpha=0.5)

oos_dataset = EcoliAntibioticResistanceDataset(antibiotic='Ciprofloxacin',
        test_size=0.20)

oos_test, _ = oos_dataset.testing_data
oos_test = tic.transform(oos_test)
oos_test = st.transform(oos_test)

oos_distribution = np.amax(clf.predict_proba(oos_test), axis=1).ravel()

plt.hist(oos_distribution, label='oos ecoli', bins=np.linspace(0.50, 0.60, 100), alpha=0.5)
plt.legend()

plt.show()
