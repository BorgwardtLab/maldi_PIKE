'''
Demo file for kernel calculation.
'''

from ismb2020_maldi.datasets import EcoliAntibioticResistanceDataset

from maldi_learn.kernels import DiffusionKernel

from maldi_learn.preprocessing import SubsetPeaksTransformer

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import average_precision_score
from sklearn.svm import SVC


import numpy as np
import sys

dataset = EcoliAntibioticResistanceDataset(antibiotic='Ceftriaxon',
        test_size=0.20)
X_train, y_train = dataset.training_data
X_test, y_test = dataset.testing_data

st = SubsetPeaksTransformer(n_peaks=100)

X_train = st.fit_transform(X_train)
X_test = st.transform(X_test)

kernel = DiffusionKernel(sigma=400)

clf = GaussianProcessClassifier(kernel=kernel, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict_proba(X_test)
average_precision = average_precision_score(y_test, y_pred[:, 1])

print(f'Average precision: {100 * average_precision:2.2f}')
