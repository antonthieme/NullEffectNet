import scanpy as sc
from statsmodels.stats.multitest import multipletests
import anndata as ad
import torch
import pandas as pd
import requests
import sys
import os
import scanpy as sc
#import pertpy as pt      # comment-out when using nen_env
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

def preprocess_data(adata, gene_ref_path, min_cells=3):
    perturbation_counts = adata.obs['gene_list'].value_counts()
    valid_gene_lists = perturbation_counts[perturbation_counts >= min_cells].index
    adata = adata[adata.obs['gene_list'].isin(valid_gene_lists)]

    gene_ref = pd.read_csv(gene_ref_path, sep='\t')
    subs = gene_ref[['Approved symbol', 'Ensembl gene ID']].dropna()
    ensembl_to_symbol = dict(zip(subs['Ensembl gene ID'], subs['Approved symbol']))

    adata.obs['perturbation_symbol'] = adata.obs['gene_list'].map(ensembl_to_symbol).fillna(adata.obs['gene_list'])
    return adata

def run_de_analysis_with_scanpy(adata):
    # Identify perturbation groups
    perturbations = adata.obs['gene_list'].unique()
    perturbations = [p for p in perturbations if (p != 'control') and (p != '')]

    results = []

    for pert in perturbations:
        print(pert)
        # Compare each perturbation with control
        sc.tl.rank_genes_groups(adata, 'gene_list', groups=[pert], reference='control')
        degs = sc.get.rank_genes_groups_df(adata, group=pert)
        degs['perturbation'] = pert
        results.append(degs)
    
    # Combine results
    result_df = pd.concat(results, ignore_index=True)
    print(result_df)
    
    # Adjust p-values for multiple testing (done internally by scanpy)
    #result_df['pvals_adj'] = multipletests(result_df['pvals'], method='fdr_bh')[1]
    
    return result_df

perturbation_screens_dir = '../data/perturbation_screens/preprocessed_all_genes'
#file_to_process = 'GasperiniShendure2019_lowMOI_saved_all_genes.h5ad'

results_list = []

for file_to_process in os.listdir(perturbation_screens_dir):

    # Load data and prepare
    try:
        adata = sc.read_h5ad(f'{perturbation_screens_dir}/{file_to_process}')
        adata = preprocess_data(adata, '../data/id_mappings/gene_ref.tsv')
    except Exception as e:
        print(f'Dataset {file_to_process} failed during loading and preprocessing with exception {e}')
        continue

    # Run DE analysis
    try:
        result_df = run_de_analysis_with_scanpy(adata)
        result_df.to_csv(f"../data/perturbation_screens/deg/{file_to_process[:file_to_process.find('_saved')]}_deg.csv")
    except Exception as e:
        print(f'Dataset {file_to_process} failed during DEG analysis with exception {e}')
        continue
    #output_file = f'../data/{perturbation_screen_name}_deg_scanpy.csv'
    #save_results(result_df, output_file)
    results_list.append(result_df)

all_results = pd.concat(results_list)
all_results.to_csv('../data/perturbation_screens/deg/deg_results_scanpy.csv')