#!/usr/bin/env python
# coding: utf-8

# Imports as per their example
import os
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Union
from matplotlib.pyplot import imread
import liana as li
import decoupler as dc
import omnipath

from gbmhackathon.utils.visium_functions import (
   normalize_anndata_wrapper,
   convert_obsm_to_adata
)
from gbmhackathon.viz.visium_functions import (
   plot_spatial_expression,
   plot_obsm
)
from gbmhackathon.stats.visium_functions import (
   perform_multi_clustering,
   quantify_cell_population_activity
)
from gbmhackathon import MosaicDataset

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 1200


# Loading the visium
visium_dict = MosaicDataset.load_visium(
   resolution="hires"
)


# # Feature extraction


## Normalize visium data
visium_obj = normalize_anndata_wrapper(visium_dict, target_sum=1e6)


## Get the metaprograms and their specific genes
metaprograms_path = "Metaprograms_10_epithMalignant_50genes_RNA_assay.csv"
metaprograms_df = pd.read_csv(metaprograms_path, index_col=0)


## Functions for calculating
import scipy

def calculate_spot_expression(
   adata: ad.AnnData,
   gene_list: List[str],
   spot_coordinates: tuple[int, int],
   radius: float = 1.0,
   layer: str = "log_CPM",
) -> Dict[str, Union[Dict[str, float], int, np.ndarray]]:
   """
   Calculate expression for specified genes at a given spatial location for a single sample.
   
   Parameters
   ----------
   adata : ad.AnnData
       AnnData object containing the spatial transcriptomics data
   gene_list : List[str]
       List of genes to analyze
   spot_coordinates : Tuple[int, int]
       (x, y) coordinates of the spot of interest
   radius : float, optional
       Radius to consider when looking for nearest spot, by default 1.0
   layer : str, optional
       Expression layer to use, by default "log_CPM"
       
   Returns
   -------
   Dict[str, Union[Dict[str, float], int, np.ndarray]]
       Dictionary containing:
       - 'spot_expression': Dictionary mapping gene names to their expression values
       - 'spot_index': Index of the identified spot
       - 'spot_coordinates': Array of [x, y] coordinates of the identified spot
   """    
   # Verify spatial coordinates exist
   if 'spatial' not in adata.obsm:
       raise ValueError("Spatial coordinates not found in the AnnData object")
       
   x, y = spot_coordinates
   
   # Find nearest spot to the given coordinates
   spatial_coords = adata.obsm['spatial']
   distances = np.sqrt(
       (spatial_coords[:, 0] - x)**2 + 
       (spatial_coords[:, 1] - y)**2
   )
   nearest_spot_idx = np.argmin(distances)
   
   # Check if the nearest spot is within the specified radius
   if distances[nearest_spot_idx] > radius:
       raise ValueError(f"No spots found within radius {radius}")
       
   # Create a list of gene indices, setting None for missing genes
   gene_indices = []
   for gene in gene_list:
       try:
           gene_indices.append(list(adata.var_names).index(gene))
       except ValueError:
           gene_indices.append(None)
   
   # Extract expression values, using 0 for missing genes
   expression_values = []
   for gene, idx in zip(gene_list, gene_indices):
       if idx is not None:
           if isinstance(adata.layers[layer], scipy.sparse.spmatrix):
               value = adata.layers[layer][nearest_spot_idx, idx].flatten()[0]
           else:
               value = adata.layers[layer][nearest_spot_idx, idx]
       else:
           value = 0.0
       expression_values.append(value)
   
   # Create dictionary of gene expressions
   gene_expression = dict(zip(gene_list, expression_values))
   
   return {
       'spot_expression': gene_expression,
       'spot_index': nearest_spot_idx,
       'spot_coordinates': spatial_coords[nearest_spot_idx]
   }

def get_mp_expression_for_spot(adata, spot_coordinates):
   mp_expression_sum = np.zeros(metaprograms_df.shape[1])
   mp_expression_avg = np.zeros(metaprograms_df.shape[1])
   for i, col in enumerate(metaprograms_df.columns):
       genes = metaprograms_df[col].dropna().values
       result = calculate_spot_expression(
           adata,
           genes,
           spot_coordinates,
           radius=1.5,
           layer='log_CPM'
       )
       values = list(result['spot_expression'].values())
       total = sum(values)
       average = total / len(values)
       mp_expression_sum[i] = total
       mp_expression_avg[i] = average

   return mp_expression_sum, mp_expression_avg


# ## Example usage

sample_id = 'HK_G_030a_vis' ## example sample id

# Example usage
expression_result_sum, expression_result_avg = get_mp_expression_for_spot(
   adata=visium_obj[sample_id],
   spot_coordinates=(14737, 17241), ## example coordinates for the visium spot
)

print(expression_result_sum)
print(expression_result_avg)
