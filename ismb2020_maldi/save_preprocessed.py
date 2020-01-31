'''
Read all spectra, preprocess them with TopologicalPeakFiltering and then save to new folder.
'''

from ismb2020_maldi.datasets import AntibioticResistanceDataset

from maldi_learn.data import write_spectra
from maldi_learn.preprocessing import TopologicalPeakFiltering


dataset = AntibioticResistanceDataset(test_size=0.5)

# write testing_data
X, y = dataset.testing_data

topf = TopologicalPeakFiltering(n_peaks=False)
X_sparse = topf.transform(X)

write_spectra(X_sparse, y, '/links/groups/borgwardt/Data/ismb2020_maldi/spectra_preprocessed')


# write training_data
X, y = dataset.training_data

topf = TopologicalPeakFiltering(n_peaks=False)
X_sparse = topf.transform(X)

write_spectra(X_sparse, y, '/links/groups/borgwardt/Data/ismb2020_maldi/spectra_preprocessed')
