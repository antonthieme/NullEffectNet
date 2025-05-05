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
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from tqdm import tqdm

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
h5ad_files = [f for f in os.listdir(perturbation_screens_dir) if f.endswith('all_genes.h5ad')]
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
log_file = f"../logs/deg/{perturbation_screen_name}_{timestamp}.log"
logger = setup_logger(log_file, args.job_id, file_to_process)

logger.info(f"Starting perturbation analysis script for file: {file_to_process}")
logger.info(f"Log file created at: {log_file}")

load_dotenv(find_dotenv())
logger.info("Environment variables loaded")

sys.path.append('../src/null-effect-net')
logger.info("Added null-effect-net to path")

import utils   

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
        
        # Further filtering based on cell counts
        perturbation_counts = adata.obs['gene_list'].value_counts()
        logger.debug(f"Perturbation counts: min={perturbation_counts.min()}, max={perturbation_counts.max()}, mean={perturbation_counts.mean():.2f}")

        min_cells = 3
        valid_gene_lists = perturbation_counts[perturbation_counts >= min_cells].index
        logger.debug(f"Valid gene lists with at least {min_cells} cells: {len(valid_gene_lists)}")

        adata = adata[adata.obs['gene_list'].isin(valid_gene_lists)]
        logger.debug(f"After min cell filtering: {adata.shape[0]} cells")

        gene_ref = pd.read_csv('../data/id_mappings/gene_ref.tsv', sep='\t')

        subs = gene_ref[['Approved symbol', 'Ensembl gene ID']]
        subs.dropna(inplace=True)

        all_ensembl = list(gene_ref['Ensembl gene ID'])
        all_gene_id = list(gene_ref['Approved symbol'])

        # Step 1: Build Ensembl to Gene Symbol mapping
        ensembl_to_symbol = dict(zip(all_ensembl, all_gene_id))

        # Step 2: Map perturbation column to gene symbols where possible
        adata.obs['perturbation_symbol'] = adata.obs['gene_list'].map(ensembl_to_symbol).fillna(adata.obs['gene_list'])

        # Step 3: Get control and perturbed groups
        control_cells = adata.obs[adata.obs['perturbation_symbol'] == 'control'].index
        perturbations = adata.obs['perturbation_symbol'].unique()
        perturbations = [p for p in perturbations if p != 'control']

        # Step 4: Run DE for each perturbation vs control
        results = []

        X_df = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
                            columns=adata.var_names, index=adata.obs_names)

        for pert in tqdm(perturbations):
            perturbed_cells = adata.obs[adata.obs['perturbation_symbol'] == pert].index

            # Get expression matrices
            control_expr = X_df.loc[control_cells]
            perturbed_expr = X_df.loc[perturbed_cells]

            # Compute logFC and p-values
            logfc = np.log2(perturbed_expr.mean(axis=0) + 1) - np.log2(control_expr.mean(axis=0) + 1)
            pvals = ttest_ind(perturbed_expr, control_expr, axis=0, equal_var=False).pvalue

            df = pd.DataFrame({
                'gene': X_df.columns,
                'perturbation': pert,
                'logFC': logfc,
                'pval': pvals
            })

            results.append(df)

        # Step 5: Combine all results
        result_df = pd.concat(results, ignore_index=True)

        logger.debug(f"Results summary: {len(result_df)} perturbations analyzed")
        
        # Save results
        output_file = f'../data/{perturbation_screen_name}_deg.csv'
        logger.info(f"Saving results to {output_file}")
        result_df.to_csv(output_file)
        
        logger.info(f"Completed analysis for {perturbation_screen_name}")

    except Exception as e:
        logger.error(f'Error processing {perturbation_screen_name}: {str(e)}', exc_info=True)


# Process the selected file
process_perturbation_screen(file_to_process)

logger.info("Script execution completed")