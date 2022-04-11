#!/usr/bin/env python
#
# Script to extract all results from one species-antibiotic scenario
# and store values in csv, to create tikzpix figures

import numpy as np
import pandas as pd
import json_tricks as jt
from tqdm import tqdm

import argparse

metrics_agg = {
    'accuracy': [np.mean, np.std, 'count'], 
    'average_precision': [np.mean, np.std],
    'davies_bouldin_score': [np.mean],
    'silhouette_score': [np.mean],
    'calinski_harabasz_score': [np.mean],
               }

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', type=str, required=True)
    parser.add_argument('-s', '--species', type=str, required=True)
    parser.add_argument('-a', '--antibiotic', type=str, required=True)
    parser.add_argument('-l', '--cluster_linkage', type=str, required=True)
    parser.add_argument('INPUT', nargs='+', type=str)
    
    args = parser.parse_args()
    
    rows = []

    for filename in tqdm(args.INPUT, desc='Loading'):
        with open(filename) as f:
            # Ensures that we can parse normal JSON files
            pos = 0            

            for line in f: 
                
                if line.startswith('{'):
                    f.seek(pos)
                    break
                else:
                    pos += len(line)

            # Check whether file is empty for some reason. 
            # If so, we can skip it.
            line = f.readline()
            if line == '':
                continue

            # Not empty, so we need to reset the file pointer
            else:
                f.seek(pos)
            try:
                data_raw = jt.load(f, ignore_comments=True)
            except:
                print(f)            

        # check whether current file is correct scenario
        if data_raw['species'] != args.species:
            continue 
        if data_raw['antibiotic'] != args.antibiotic:
            continue 
        if data_raw['linkage'] != args.cluster_linkage:
            continue 

        is_baseline = 'best_parameters' in data_raw
        method = 'baseline' if is_baseline else 'kernel'

        #has_featsel = 'feature_selection' in data_raw
        #method = method + '_featsel' if has_featsel else method
        
        if method != args.method:
            continue
        
        # Create new row for the current json file    
        row = {
          'species': data_raw['species'],
          'antibiotic': data_raw['antibiotic'],
          'linkage': data_raw['linkage'],
          'n_cluster': data_raw['n_cluster'],
            }
        
        
        for m in metrics_agg.keys(): 
            if data_raw.get(m) is not None:
                row[m] = data_raw[m] 

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.groupby(['species', 'antibiotic', 'linkage', 'n_cluster']).agg(metrics_agg)   
   
    df.reset_index(inplace=True)
    df.columns = df.columns.to_flat_index()
    df.columns = ['_'.join(col) if col[1]!='' else str(col[0]) for col in df.columns] 
    
    if np.any(df['accuracy_count'].values != 5):
        print('less than 5 reps') 
        print(df[df.columns[:7]]) 
    elif df.shape[0] != 20:
        print('entire rows are missing')    
        print(df[df.columns[:7]]) 
    else:
        print(f'./results/results_DRIAMS/clustering_results_csv/{args.method}_{args.antibiotic}_{args.species}_{args.cluster_linkage}.csv is complete') 

    df.to_csv(f'./results/results_DRIAMS/clustering_results_csv/{args.method}_{args.antibiotic}_{args.species}_{args.cluster_linkage}.csv')   
