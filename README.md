# NullEffectNet

A deep learning framework for predicting gene perturbation effects using different neural network architectures, pre-trained embeddings and biological networks.

## Overview

NullEffectNet is designed to predict the effect significance of CRISPR perturbations of different target genes. The framework supports multiple model architectures:

- MLP (Multi-Layer Perceptron)
- GNN (Graph Neural Network)
- GNN with Attention mechanisms

The framework can be used with different pre-trained embeddings of genes. Currently implemented are:

- ESM
- SubCell
- PINNACLE

The framework can additionally be used with different biological networks. Currently implemented is only a protein-protein interaction based network.

## Environment setup with Conda etc.

    conda env create -f environment.yml
    conda activate nen_env

## Notebooks

### Preprocessing
Jupyter notebooks for preprocessing embeddings, perturbation screen data, and the protein-protein interaction network, (as well as cell type specific expression (experimental)) can be found in the notebooks folder with prefix "001".

The final step of preprocessing involves compiling all preprocessed datasets into training and test data. This Jupyter notebook can be found in the notebooks folder with prefix "002".

### Model development
Models were developed using the Jupyter notebooks with the prefix "003".

### Experiments
Experiments were performed using the Jupyter notebooks with the prefix "004".

## Scripts

In the scripts folder, python and slurm batch scripts for large preprocessing jobs can be found.

## Src

The core code of NullEffectNet can be found in in src/null-effect-net directory and is modularized in the following way:
- models: pytorch model classes
- dataset: dataset classes for each model and their collate function
- train_utils: utility functions for training and evaluating models
- utils: general utility functions mostly used during preprocessing