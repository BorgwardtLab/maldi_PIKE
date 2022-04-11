"""Dataset of MALDI-TOF spectra for antibiotic resistance prediction."""
import os

from dotenv import load_dotenv
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import homogeneity_score
import random
from sklearn.model_selection import train_test_split

from maldi_learn.data import MaldiTofSpectrum
from sklearn.feature_selection import VarianceThreshold
from maldi_learn.vectorization import BinningVectorizer
from .utilities import load_data, split_data

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score

from .dataset import Dataset

# Dataset paths are specified in .env file in the root of the repository
load_dotenv("/cluster/home/sebalzer/maldi_thesis/ismb2020_maldi/config.env")
# load_dotenv("/links/groups/borgwardt/Data/maldi_student/ismb2020_maldi/config.env")
DRIAMS_ROOT = os.getenv('DRIAMS_ROOT_PATH')
CLUSTER_DATA_DIR = os.getenv('CLUSTER_DATA_DIR') # data used for clustering 
CLASSIFICATION_DATA_DIR = os.getenv('CLASSIFICATION_DATA_DIR') # data used for classification

site = 'DRIAMS-A'
years = ['2017', '2018']

class AntibioticResistanceDataset(Dataset):
    """Base class of datasets predicting antibiotic resistance."""

    #endpoint_file_name = 'IDRES_clean.csv'

    def __init__(self, antibiotic, test_size=0.2, random_seed=2020,
            suffix='', cutoff=None):
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
        self.cutoff = cutoff

        all_instances = self._make_binary_labels(
            self._read_endpoints_and_preprocess())
        self.all_instances = all_instances
        
        if (cutoff is not None):
            stratify_by = self._cluster()
            
        else:
            stratify_by = all_instances.values

        train_instances, test_instances = train_test_split(
            all_instances,
            test_size=test_size,
            random_state=random_seed,
            stratify=stratify_by  # stratify by labels (and optionally clusters)
        )

        self.train_instances, self.test_instances = train_instances, test_instances
            
    def _cluster(self):
        """
        Clusters the entire dataset
        1-sample-clusters raise an error in train_test_split and are therefore
        added to the nearest cluster

        Returns
        -------
        DataFrame with shape [n_samples x 2]
        with labels in first col and cluster labels in second col
        """
        # Bin spectra
        bv = BinningVectorizer(n_bins=6000, min_bin=2000, max_bin=20000)
        spectra, instances = self._read_data(self.all_instances, path="")
        bv.fit(spectra)
        X = bv.transform(spectra)
        y = self.all_instances.values
        
        # Binarize and select features
        X[X != 0] = 1
        sel = VarianceThreshold(threshold=0.98*(1-0.98))
        X = sel.fit_transform(X)
        
        # Perform clustering
        linked = linkage(X, 'ward')
        cluster_labels = fcluster(linked, t=self.cutoff, criterion='maxclust')
        homogeneity = homogeneity_score(y, cluster_labels)
        print(f"Homogeneity score before merging: {homogeneity}")
        
        # Add 1-sample-clusters to closest cluster, keep track of how many
        unique_labels, cluster_sizes = np.unique(cluster_labels, return_counts=True)
        print("Number of clusters before merging:", len(unique_labels))
        counter = 0
        
        for size, label in zip(cluster_sizes, unique_labels):
            if size == 1:
                counter = counter + 1
                index = np.where(cluster_labels == label)[0][0]
                cluster_labels[index] = label + 1 if label < unique_labels.max() else label - 1
                unique_labels, cluster_sizes = np.unique(cluster_labels, 
                                                         return_counts=True)
        print(f"Number of 1-sample-clusters: {counter}")
        
        # Check for stratification classes occuring only once
        stratify_by = np.concatenate((y[:, np.newaxis], 
                                      cluster_labels[:, np.newaxis]), axis=1)
        strat_classes, sizes = np.unique(stratify_by, return_counts=True, axis=0)
        counter = 0
        
        while sizes.min() < 2:
            for size, label in zip(sizes, strat_classes):
                if size == 1:
                    counter = counter + 1
                    cl_label = label[1]
                    index = np.where(cluster_labels == cl_label)[0]
                    cluster_labels[index] = cl_label + 1 if cl_label < unique_labels.max() else cl_label - 1
                    unique_labels, cluster_sizes = np.unique(cluster_labels, 
                                                             return_counts=True)
                    stratify_by = np.concatenate((y[:, np.newaxis], 
                                          cluster_labels[:, np.newaxis]), axis=1)
                    strat_classes, sizes = np.unique(stratify_by, return_counts=True, axis=0)
        print(f"Number of strat classes of size 1: {counter}")
        
        # Determine cluster info
        homogeneity = homogeneity_score(y, cluster_labels)
        print(f"Clustering done, homogeneity score is {homogeneity:3.2}")
        print("Number of clusters after merging:", len(unique_labels))
        print("Samples per cluster:", 
              f"{cluster_sizes.mean():4.3} +- {cluster_sizes.std():4.3}")
        resistant_ratio = np.full(len(unique_labels), None)
        for i, label in enumerate(unique_labels, 0):
            indices = np.where(cluster_labels == label)[0]
            resistant_ratio[i] = round(y[indices].sum() / len(indices), 2)
        print(f"Percentage of resistant samples for each cluster: {resistant_ratio}")
        
        stratify_by = np.concatenate((y[:, np.newaxis], 
                                      cluster_labels[:, np.newaxis]), axis=1)
        
        return stratify_by

    def _read_endpoints_and_preprocess(self):
        print("endpoint path: ", "", "\n")
        endpoint_file = os.path.join("", self.endpoint_file_name)
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
    def _build_filepaths_from_codes(codes, suffix, path=""):
        return [os.path.join(path, f'{code}{suffix}.txt') for code in codes]

    def _read_spectra(self, files):
        return [
            MaldiTofSpectrum(
                pd.read_csv(f, sep=' ', comment='#', engine='c').values)
            for f in files
        ]

    def _read_data(self, instances, path=""):
        codes = instances.index
        files = self._build_filepaths_from_codes(codes, self.suffix, path)
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


class ClusterAntibioticResistanceDataset(AntibioticResistanceDataset):
    """Dataset for antibiotic resistance with clustering option."""

    def __init__(self, 
                 species,
                 antibiotic, 
                 test_size=0.2, 
                 cluster_based_split=True, 
                 cluster_feature_selection=True, 
                 bv=BinningVectorizer(n_bins=6000),
                 n_cluster=2,
                 cluster_linkage='ward',   
                 random_seed=2020, 
                 suffix=''
        ):
        """Initialize the dataset.

        Args:
            antibiotic:     Name (str) of the antibiotic to use for
                            generating labels (endpoints).
            test_size:      Fraction of the data that should be returned for
                            testing.
            cluster_based_split: Boolean indicating whether a clustering 
                            should be perfomed and used for stratifying 
                            the train-test split.
            bv:             Define BinningVectorizer used to bin peaks for 
                            clustering. 
            n_cluster:      Number of clusters which will be determined through
                            hierarchical clustering. 
            cluster_linkage: Clustering linkage used in hierarchical 
                            clustering.
            random_seed:    Random seed for splitting the data into train and
                            test.
            suffix:         Suffix to use for the files to load. This suffix
                            will be appended to the code specified in the 
                            endpoints data file.
        """

        self.species = species
        self.antibiotic = antibiotic
        self.suffix = suffix
        self.fs = cluster_feature_selection
        self.linkage = cluster_linkage
        self.bv = bv
        self.n_cluster = n_cluster

        # Load data for clustering in case it is requested
        if cluster_based_split and (n_cluster > 1):
            self.cluster_data = load_data(
                root=DRIAMS_ROOT,
                site=site,
                years=years,
                species=species,
                antibiotic=antibiotic,
                spectra_type=CLUSTER_DATA_DIR)
        else:
            self.cluster_data = None

        # Load data for classification, if it was not already loaded for clustering
        if (CLASSIFICATION_DATA_DIR != CLUSTER_DATA_DIR) or (self.cluster_data is None):
            self.classification_data = load_data(
                root=DRIAMS_ROOT,
                site=site,
                years=years,
                species=species,
                antibiotic=antibiotic,
                spectra_type=CLASSIFICATION_DATA_DIR)
        else:
            self.classification_data = self.cluster_data
        
        # Only use cluster based stratification in case it is requested
        if not cluster_based_split:
            class_cluster_labels = None # case based stratification only
        elif n_cluster==1:
            class_cluster_labels = None # case based stratification only
        else:
            class_cluster_labels = self._get_stratification_labels() 

        self.X_train, self.y_train, self.X_test, self.y_test = split_data(
            antibiotic,
            test_size,
            random_seed,
            self.classification_data.X,
            self.classification_data.y,
            class_cluster_labels
        )

    
    def _get_stratification_labels(self):

        class_labels = self.cluster_data.y[self.antibiotic].to_numpy(dtype='int32')
        
        # get cluster labels
        cluster_labels = self._cluster()
        assert isinstance(class_labels, np.ndarray)
        assert isinstance(cluster_labels, np.ndarray)

        stratification_labels = np.column_stack(
                            (class_labels, 
                             cluster_labels)
                            )

        #some logic to deal with single (class, cluster) samples
        arrays, counts = np.unique(stratification_labels, axis=0, return_counts=True)

        # extract unique class and cluster labels as tuples
        single_arrays = arrays[counts==1,:]
        single_tuples = set()
        [single_tuples.add(tuple(row)) for row in single_arrays]

        # if there exists more then one unique tuple create
        if len(single_tuples) == 1: 
            for i, row in enumerate(stratification_labels):
                if tuple(row) in single_tuples:
                    indices_without_i = list(np.arange(i)) + list(np.arange(i+1,stratification_labels.shape[0]))
                    stratification_labels[i,:] = stratification_labels[random.choice(indices_without_i),:]

        elif len(single_tuples) > 1:
            for i, row in enumerate(stratification_labels):
                if tuple(row) in single_tuples:
                    stratification_labels[i,:] = [999, 999]

        return stratification_labels
    

    def _cluster(self):

        # read in all instances
        peaks = self.cluster_data.X

        # spectra binning
        X = self.bv.fit_transform(peaks)
       
        # binarization and feature selection 
        if self.fs:
            X[X != 0] = 1
            fs = VarianceThreshold(threshold=0.98*(1-0.98)) 
            X = fs.fit_transform(X)  
        
        # clustering
        l = linkage(X, self.linkage)
        
        # plot and save dendrogram
        # plt.figure(figsize = (80, 50))
        # dendrogram(l, color_threshold=90, no_labels=True)
        # plt.savefig(f"dendrogram_{self.species}.pdf", bbox_inches="tight", pad_inches=1)
        # print("Dendrogram saved. \n")

        cluster_labels = fcluster(l, t=self.n_cluster, criterion='maxclust') 

        # calculate clustering validity indices if possible
        try: 
            self.davies_bouldin_score = davies_bouldin_score(X, cluster_labels)
            self.silhouette_score = silhouette_score(X, cluster_labels)
            self.calinski_harabasz_score = calinski_harabasz_score(X, cluster_labels)
        except ValueError: 
            print("Clustering validity indices could not be calculated. \n \
                In most cases, only a single cluster could be formed for \n \
                the provided linkage criterion.")

        return cluster_labels
