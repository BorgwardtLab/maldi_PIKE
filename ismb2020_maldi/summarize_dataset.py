'''
Provide a summary over the dataset, including class balance for each species and antibiotic.
'''

from ismb2020_maldi.datasets import EcoliAntibioticResistanceDataset, SaureusAntibioticResistanceDataset, KpneuAntibioticResistanceDataset

import numpy as np
import sys


datasets_map = {
    'ecoli': EcoliAntibioticResistanceDataset,
    'saureus': SaureusAntibioticResistanceDataset,
    'kpneu': KpneuAntibioticResistanceDataset
}

antibiotic_map = {
    'ecoli': ['Ciprofloxacin', 'Ceftriaxon','Amoxicillin-Clavulansaeure'],
    'saureus': ['Ciprofloxacin', 'Penicillin','Amoxicillin-Clavulansaeure'],
    'kpneu': ['Ciprofloxacin', 'Ceftriaxon','Piperacillin-Tazobactam'] 
}

for species in datasets_map.keys():
   
    print(f'\n{species}')     
    Dataset = datasets_map[species]
    for antibiotic in antibiotic_map[species]:
        
        print(f'{antibiotic}')
        dataset = Dataset(antibiotic, test_size=0.2)
        _, y_complete = dataset.complete_data
        _, y_train = dataset.training_data
        _, y_test = dataset.testing_data

        #for y in [y_complete, y_train, y_test]:
        for y in [y_complete]:
            counts = y.value_counts()
            print(counts)
            print(y.shape[0])
            assert counts.loc[0]+counts.loc[1] == y.shape[0]
            print(round(counts.loc[1]/float(y.shape[0]), 3))
        print()
