"""Dataset of MALDI-TOF spectra for antibiotic resistance prediction."""
import os

from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
from maldi_learn.data import MaldiTofSpectrum

from .dataset import Dataset

# Dataset paths are specified in .env file in the root of the repository
load_dotenv()
SPECTRA_PATH = os.getenv('ANTIBIOTICS_SPECTRA_PATH')
ENDPOINT_PATH = os.getenv('ANTIBIOTICS_ENDPOINT_PATH')


class AntibioticResistanceDataset(Dataset):
    """Base class of datasets predicting antibiotic resistance."""

    # endpoint_file_name = 'IDRES_clean.csv'

    def __init__(self, antibiotic, test_size=0.2, random_seed=2020,
            suffix=''):
        """Initialize the dataset.

        Args:
            antibiotic: Name (str) of the antibiotic to use for
            generating labels (endpoints).
            test_size: Fraction of the data that should be returned for
                testing.
            random_seed: Random seed for splitting the data into train and
                test.
            suffix: Suffix to use for the files to load. This suffix
            will be appended to the code specified in the endpoints data
            file.
        """
        self.antibiotic = antibiotic
        self.suffix = suffix

        all_instances = self._make_binary_labels(
            self._read_endpoints_and_preprocess())

        train_instances, test_instances = train_test_split(
            all_instances,
            test_size=test_size,
            random_state=random_seed,
            stratify=all_instances.values  # stratify by labels
        )

        self.all_instances, self.train_instances, self.test_instances = \
            all_instances, train_instances, test_instances

    def _read_endpoints_and_preprocess(self):
        endpoint_file = os.path.join(ENDPOINT_PATH, self.endpoint_file_name)
        endpoints = pd.read_csv(endpoint_file, index_col='code')
        endpoints = endpoints.replace({
            '-': float('NaN'),
            'R(1)': float('NaN'),
            'L(1)': float('NaN'),
            'I(1)': float('NaN'),
            'I(1), S(1)': float('NaN'),
            'R(1), I(1)': float('NaN'),
            'R(1), S(1)': float('NaN'),
            'R(1), I(1), S(1)': float('NaN')
        })

        return endpoints

    def _make_binary_labels(self, df):
        """
        Creates binary labels by restricting the input data frame to the
        specified antibiotic. This is followed by dropping all NaNs, and
        making all labels binary (depending on resistance/susceptibility).
        """

        only_antibiotic = df[self.antibiotic]

        only_antibiotic = only_antibiotic.dropna(
            axis='index', how='any', inplace=False)

        return only_antibiotic.replace({'R': 1, 'I': 1, 'S': 0})

    # TODO: might want to remove this
    def _subset_instances(self, *instance_lists):
        def subset_and_binarize(input_instances):
            """Remove unused antibiotics and not measured instances."""
            only_antibiotic = input_instances[self.antibiotic]
            only_antibiotic = only_antibiotic.dropna(axis='index', how='any', inplace=False)
            return only_antibiotic.replace({'R': 1, 'I': 1, 'S': 0})
        return [subset_and_binarize(instances) for instances in instance_lists]

    @staticmethod
    def _build_filepaths_from_codes(codes, suffix):
        return [os.path.join(SPECTRA_PATH, f'{code}{suffix}.txt') for code in codes]

    def _read_spectra(self, files):
        return [
            MaldiTofSpectrum(
                pd.read_csv(f, sep=' ', comment='#', engine='c').values)
            for f in files
        ]

    def _read_data(self, instances):
        codes = instances.index
        files = self._build_filepaths_from_codes(codes, self.suffix)
        spectra = self._read_spectra(files)
        return spectra, instances

    @property
    def training_data(self):
        """Get spectra used for training."""
        return self._read_data(self.train_instances)

    @property
    def validation_data(self):
        """Not implemented for now."""
        raise NotImplementedError()

    @property
    def testing_data(self):
        """Get spectra used for testing."""
        return self._read_data(self.test_instances)

    @property
    def complete_data(self):
        """Get all spectra."""
        return self._read_data(self.all_instances)


class EcoliAntibioticResistanceDataset(AntibioticResistanceDataset):
    """Dataset for E.coli antibiotic resistance."""

    endpoint_file_name = 'IDRES_Ecoli.csv'


class SaureusAntibioticResistanceDataset(AntibioticResistanceDataset):
    """Dataset for S.aureus antibiotic resistance."""

    endpoint_file_name = 'IDRES_Saureus.csv'


class KpneuAntibioticResistanceDataset(AntibioticResistanceDataset):
    """Dataset for K.pneumoniae antibiotic resistance."""

    endpoint_file_name = 'IDRES_Kpneu.csv'
