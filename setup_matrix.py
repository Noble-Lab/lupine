"""
SETUP-MATRIX

Load in the pre-fitted model weights, recreate the Lupine-imputed
joint protein quantifications matrix, then attach the samples/runs
provided by the user to the existing training matrix. 

It would be great if this script didn't require a GPU. In practice
the matrix reconstruction step might just be too slow if all of the
tensors are loaded on CPU. 

Would also be great if this were a module within the `lupine` package
as opposed to a standalone script.  
"""
import pandas as pd
import numpy as np
import torch
import click 
from lupine.data_loaders import FactorizationDataset

from lupine.os_utils import os
from pathlib import Path

# @click.command()
# @click.option("--csv", required=True, nargs=1, type=str,
# 	help="Path to a CSV containing the runs to be imputed")

# The relative path to the pre-baked Lupine model 
model_path="/net/noble/vol2/home/lincolnh/code/lupine/scratch/OPT_MODEL_INTERNAL.pt"

print("\nLoading the model...")
# lupine_fitted = torch.load(model_path, map_location="cpu")
lupine_fitted = torch.load(model_path)

# Init an (empty) reconstructed matrix
n_prots = lupine_fitted.prot_factors.shape[0]
n_runs = lupine_fitted.run_factors.shape[1]

X = np.zeros((n_prots, n_runs))
X[:] = np.nan

# Convert to a torch tensor; load on CUDA
X = torch.tensor(X, device="cuda")

# Init a data loader
loader = FactorizationDataset(
			X, 
			X_val=None,
			partition="Train",
			batch_size=128, 
			biased=False,
			shuffle=True, 
			missing=True,
			testing=False,
			rand_seed=42,
)
loader.get_standard_loader()

# Use the fitted model to impute. This step is absurdly slow if 
#   the tensors aren't loaded on CUDA. 
print("Generating the training matrix...")
with torch.no_grad():
	for locs, _ in loader:
		target = X[tuple(locs.T)]
		X[tuple(locs.T)] = lupine_fitted(locs).type_as(target)

# Inverse scale. This should use the train set scaling factor
X = lupine_fitted.scaler.inverse_transform(X, "Eval")
X = X.detach().cpu().numpy()

ensp_ids = pd.read_csv("joint_quants_ENSP_IDs.csv", index_col=0)
sample_ids = pd.read_csv("joint_quants_sample_IDs.csv", index_col=0)
ensp_ids = list(ensp_ids["0"])
sample_ids = list(sample_ids["0"])

recon_mat = pd.DataFrame(X, columns=ensp_ids, index=sample_ids)

print(recon_mat)

print("done!\n")
