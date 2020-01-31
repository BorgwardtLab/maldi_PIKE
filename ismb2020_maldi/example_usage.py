"""Example usage file."""

from ismb2020_maldi.datasets import EcoliAntibioticResistanceDataset
from maldi_learn.preprocessing import TopologicalPeakFiltering
dataset = EcoliAntibioticResistanceDataset('Ciprofloxacin')
X, y = dataset.complete_data

topf = TopologicalPeakFiltering(n_peaks=100)
X_sparse = topf.transform(X)
