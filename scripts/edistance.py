import anndata as ad
import torch
import pandas as pd
import requests
import sys
import os
import scanpy as sc
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import pickle
from scipy.stats import wasserstein_distance, permutation_test
from scipy.spatial.distance import cdist, pdist, squareform

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

sys.path.append('../src/null-effect-net')

import utils   


def e_test_fast(X, Y, n_permutations=500, random_state=None):
    rng = np.random.default_rng(random_state)

    combined = np.vstack([X, Y])
    labels = np.array([0]*len(X) + [1]*len(Y))

    # Precompute full pairwise distance matrix once
    dist_mat = squareform(pdist(combined, metric='euclidean'))

    # Helper function to compute E-distance
    def compute_e_stat(group1_indices, group2_indices):
        d_xy = np.mean(dist_mat[np.ix_(group1_indices, group2_indices)])
        d_xx = np.mean(dist_mat[np.ix_(group1_indices, group1_indices)])
        d_yy = np.mean(dist_mat[np.ix_(group2_indices, group2_indices)])
        return 2 * d_xy - d_xx - d_yy

    # Real E-distance
    group1 = np.where(labels == 0)[0]
    group2 = np.where(labels == 1)[0]
    observed = compute_e_stat(group1, group2)

    # Permutation
    permutation_stats = []
    for _ in range(n_permutations):
        permuted_labels = rng.permutation(labels)
        group1_perm = np.where(permuted_labels == 0)[0]
        group2_perm = np.where(permuted_labels == 1)[0]
        permuted_stat = compute_e_stat(group1_perm, group2_perm)
        permutation_stats.append(permuted_stat)

    permutation_stats = np.array(permutation_stats)
    p_value = np.mean(permutation_stats >= observed)

    return observed, p_value


for perturbation_screen_name in os.listdir('../data/perturbation_screens'):

    if perturbation_screen_name.endswith('.h5ad'):

        perturbation_screen_name = perturbation_screen_name[:perturbation_screen_name.find('.')]

        print(f'Processing: {perturbation_screen_name}')

        try:
            print('staring preprocessing')
            adata = sc.read_h5ad(f'../data/perturbation_screens/{perturbation_screen_name}.h5ad')
            sc.pp.filter_cells(adata, min_genes=200)
            sc.pp.filter_genes(adata, min_cells=3)
            valid_indices = adata.obs[(~adata.obs['perturbation'].str.contains('_')) & (~adata.obs['perturbation'].str.contains('nan'))].index
            adata = adata[valid_indices, :]

            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

            adata.obs[adata.obs['perturbation'] != 'control']
            adata.obs['gene_list'] = adata.obs['perturbation'].apply(utils.convert_genomic_location_to_ensembl_ids)
            valid_indices = adata.obs[(~adata.obs['gene_list'].str.contains('_')) & (adata.obs['gene_list'] != 'None')].index
            adata = adata[valid_indices, :]

            print('finished preprocessing')

            adata.write(f"../data/{perturbation_screen_name}_saved_all_genes.h5ad", compression='gzip')

            perturbation_counts = adata.obs['gene_list'].value_counts()

            min_cells = 3
            valid_gene_lists = perturbation_counts[perturbation_counts >= min_cells].index

            adata = adata[adata.obs['gene_list'].isin(valid_gene_lists)]

            sc.tl.pca(adata, n_comps=50)

            results = []

            # get features (for example, 50 PCA components)
            features = adata.obsm['X_pca'][:, :50]

            # get control group
            control_mask = adata.obs['gene_list'] == 'control'
            X_control = features[control_mask]


            def run_edistance(adata, X_control, features):

                # loop through each perturbation (except 'control')
                for perturbation in adata.obs['gene_list'].unique():

                    print(perturbation)
                    
                    if perturbation == 'control':
                        continue

                    perturbation_mask = adata.obs['gene_list'] == perturbation
                    X_perturb = features[perturbation_mask]

                    if len(X_perturb) < 5 or len(X_control) < 5:

                        print(f'Skipping because less than 5 cells: {perturbation}')
                        # Skip if not enough cells
                        continue

                    edist, pval = e_test_fast(X_perturb, X_control, n_permutations=500, random_state=42)
                    results.append({'perturbation': perturbation, 'e_distance': edist, 'p_value': pval})

                return results

            results = run_edistance(adata, X_control, features)

            # Convert results to dataframe
            results_df = pd.DataFrame(results)

            results_df.to_csv(f'../data/perturbation_screens/{perturbation_screen_name}_e_distances.csv')

        except Exception as e:
            print(f'Something went wrong with {perturbation_screen_name}:   {e}')
            continue