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
import logging
import time
import argparse
from datetime import datetime
from scipy.stats import wasserstein_distance, permutation_test
from scipy.spatial.distance import cdist, pdist, squareform

from dotenv import find_dotenv, load_dotenv

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process a specific perturbation screen file')
parser.add_argument('--job_id', type=int, help='Array job ID (0-based index for file selection)')
parser.add_argument('--file_name', type=str, help='Specific h5ad file name to process (alternative to job_id)')
args = parser.parse_args()

# Set up logging
def setup_logger(log_file=None, job_id=None, file_name=None):
    """Set up logger with specified formatting."""
    # Create identifier for the log
    if job_id is not None:
        identifier = f"job_{job_id}"
    elif file_name is not None:
        identifier = file_name.replace('.h5ad', '')
    else:
        identifier = "unknown"
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file provided)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create log directory if it doesn't exist
os.makedirs("../logs", exist_ok=True)

# Determine which file to process
perturbation_screens_dir = '../data'
h5ad_files = [f for f in os.listdir(perturbation_screens_dir) if f.endswith('.h5ad')]
h5ad_files.sort()  # Sort files to ensure consistent ordering

if args.job_id is not None:
    if args.job_id < 0 or args.job_id >= len(h5ad_files):
        print(f"Error: job_id must be between 0 and {len(h5ad_files)-1}")
        sys.exit(1)
    file_to_process = h5ad_files[args.job_id]
elif args.file_name is not None:
    if args.file_name not in h5ad_files:
        print(f"Error: file '{args.file_name}' not found in {perturbation_screens_dir}")
        sys.exit(1)
    file_to_process = args.file_name
else:
    print("Error: Either --job_id or --file_name must be provided")
    print(f"Available files (0-{len(h5ad_files)-1}):")
    for i, f in enumerate(h5ad_files):
        print(f"  {i}: {f}")
    sys.exit(1)

# Set up the logger with appropriate file name
perturbation_screen_name = file_to_process[:file_to_process.find('.')]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"../logs/run_edistance/{perturbation_screen_name}_{timestamp}.log"
logger = setup_logger(log_file, args.job_id, file_to_process)

logger.info(f"Starting perturbation analysis script for file: {file_to_process}")
logger.info(f"Log file created at: {log_file}")

load_dotenv(find_dotenv())
logger.info("Environment variables loaded")

sys.path.append('../src/null-effect-net')
logger.info("Added null-effect-net to path")

import utils   

def e_test_fast(X1, X2, n_permutations=500, random_state=None):
    rng = np.random.default_rng(random_state)

    n1, n2 = len(X1), len(X2)
    X = np.vstack([X1, X2])
    labels = np.array([0]*n1 + [1]*n2)

    # Precompute all pairwise distances
    D = cdist(X, X, metric='euclidean')
    
    # Compute observed E-distance
    D_11 = D[:n1, :n1]
    D_22 = D[n1:, n1:]
    D_12 = D[:n1, n1:]
    e_obs = 2 * np.mean(D_12) - np.mean(D_11) - np.mean(D_22)

    # Permutation test
    e_null = np.empty(n_permutations)
    for i in range(n_permutations):
        rng.shuffle(labels)
        idx1 = labels == 0
        idx2 = labels == 1
        D_11 = D[np.ix_(idx1, idx1)]
        D_22 = D[np.ix_(idx2, idx2)]
        D_12 = D[np.ix_(idx1, idx2)]
        e_null[i] = 2 * np.mean(D_12) - np.mean(D_11) - np.mean(D_22)

    pval = (np.sum(e_null >= e_obs) + 1) / (n_permutations + 1)
    return e_obs, pval


def process_perturbation_screen(perturbation_screen_file):
    perturbation_screen_name = perturbation_screen_file[:perturbation_screen_file.find('.')]
    
    logger.info(f'Processing: {perturbation_screen_name}')

    try:
        logger.info('Starting preprocessing')
        
        # Load data
        file_path = f'{perturbation_screens_dir}/{perturbation_screen_file}'
        logger.debug(f"Loading dataset from {file_path}")
        adata = sc.read_h5ad(file_path)
        logger.debug(f"Dataset loaded with {adata.shape[0]} cells and {adata.shape[1]} genes")
        
        # Filter cells and genes
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        logger.debug(f"After filtering: {adata.shape[0]} cells and {adata.shape[1]} genes")
        
        # Filter for valid perturbations
        valid_indices = adata.obs[(~adata.obs['perturbation'].str.contains('_')) & (~adata.obs['perturbation'].str.contains('nan'))].index
        adata = adata[valid_indices, :]
        logger.debug(f"After perturbation filtering: {adata.shape[0]} cells")

        # Normalize
        logger.debug("Normalizing data")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Convert perturbations to gene lists
        logger.debug("Converting genomic locations to Ensembl IDs")
        adata.obs['gene_list'] = adata.obs['perturbation'].apply(utils.convert_genomic_location_to_ensembl_ids)
        valid_indices = adata.obs[(~adata.obs['gene_list'].str.contains('_')) & (adata.obs['gene_list'] != 'None')].index
        adata = adata[valid_indices, :]
        logger.debug(f"After gene list filtering: {adata.shape[0]} cells")

        logger.info('Finished preprocessing')

        # Save data
        output_path = f"../data/{perturbation_screen_name}_saved_all_genes.h5ad"
        logger.info(f"Saving processed data to {output_path}")
        adata.write(output_path, compression='gzip')

        # Further filtering based on cell counts
        perturbation_counts = adata.obs['gene_list'].value_counts()
        logger.debug(f"Perturbation counts: min={perturbation_counts.min()}, max={perturbation_counts.max()}, mean={perturbation_counts.mean():.2f}")

        min_cells = 3
        valid_gene_lists = perturbation_counts[perturbation_counts >= min_cells].index
        logger.debug(f"Valid gene lists with at least {min_cells} cells: {len(valid_gene_lists)}")

        adata = adata[adata.obs['gene_list'].isin(valid_gene_lists)]
        logger.debug(f"After min cell filtering: {adata.shape[0]} cells")

        # Run PCA
        logger.info("Running PCA")
        sc.tl.pca(adata, n_comps=50)

        results = []

        # get features (for example, 50 PCA components)
        features = adata.obsm['X_pca'][:, :50]
        logger.debug(f"Using {features.shape[1]} PCA components for analysis")

        # get control group
        control_mask = adata.obs['gene_list'] == 'control'
        X_control = features[control_mask]
        logger.info(f"Control group has {len(X_control)} cells")

        if len(X_control) < 5:
            logger.error(f"Control group has only {len(X_control)} cells, which is insufficient for analysis (minimum 5)")
            return

        logger.info("Starting E-distance calculations")
        total_perturbations = len(adata.obs['gene_list'].unique()) - 1  # Minus control
        processed = 0
        skipped = 0

        # Subsample the control group if it's very large
        MAX_CONTROL_CELLS = 1000  # Adjust this parameter based on your computational resources
        if len(X_control) > MAX_CONTROL_CELLS:
            logger.info(f"Control group is large ({len(X_control)} cells). Subsampling to {MAX_CONTROL_CELLS} cells...")
            np.random.seed(42)  # For reproducibility
            control_indices = np.random.choice(len(X_control), MAX_CONTROL_CELLS, replace=False)
            X_control_subsampled = X_control[control_indices]
        else:
            X_control_subsampled = X_control
    
        logger.info(f"Using control group with {len(X_control_subsampled)} cells")
        
        # loop through each perturbation (except 'control')
        results = []
        save_interval = 10
        result_file_partial = f'../data/perturbation_screens/{perturbation_screen_name}_partial_results.csv'

        for i, perturbation in enumerate(adata.obs['gene_list'].unique()):
            if perturbation == 'control':
                continue

            logger.debug(f"Processing perturbation {i+1}: {perturbation}")
            processed += 1

            perturbation_mask = adata.obs['gene_list'] == perturbation
            X_perturb = features[perturbation_mask]

            if len(X_perturb) < 5:
                logger.warning(f'Skipping {perturbation} (fewer than 5 cells)')
                skipped += 1
                continue

            edist, pval = e_test_fast(X_perturb, X_control_subsampled, n_permutations=500, random_state=42)
            results.append({'perturbation': perturbation, 'e_distance': edist, 'p_value': pval})

            # Save intermediate results every 10 perturbations
            if processed % save_interval == 0 or processed == total_perturbations:
                partial_df = pd.DataFrame(results)
                partial_df.to_csv(result_file_partial, index=False)
                logger.info(f"Intermediate results saved after {processed} perturbations")

        logger.info(f"E-distance calculation completed for {processed-skipped} perturbations. Skipped {skipped} perturbations.")

        # Convert results to dataframe
        results_df = pd.DataFrame(results)
        logger.debug(f"Results summary: {len(results_df)} perturbations analyzed")
        
        # Save results
        output_file = f'../data/perturbation_screens/{perturbation_screen_name}_e_distances.csv'
        logger.info(f"Saving results to {output_file}")
        results_df.to_csv(output_file)
        
        logger.info(f"Completed analysis for {perturbation_screen_name}")

    except Exception as e:
        logger.error(f'Error processing {perturbation_screen_name}: {str(e)}', exc_info=True)


# Process the selected file
process_perturbation_screen(file_to_process)

logger.info("Script execution completed")