"""
LUPINE
8.4.24

This modules contains the `Lupine` class and the implementation of
the `impute` command. `Lupine` is the high-level implementation for
a PyTorch model for imputing protein-level quantifications using
deep matrix factorization. Missing values are imputed by taking the
concatenation of the corresponding protein and run factors and 
feeding them through a deep neural network.
"""
from lupine.lupine_base import LupineBase
import torch
import click
import pandas as pd
import numpy as np
import torch

from lupine.os_utils import os
from pathlib import Path

class Lupine(LupineBase):
	"""
	A deep neural network-based matrix factorization imputation 
	model. Protein and run embeddings are randomly initialized,
	then refined through SGD. The forward pass is specified here,
	but the `LupineBase` class does all of the heavy lifting.

	Parameters
	----------
	n_prots : int, 
		The number of proteins in the quants matrix
	n_runs : int, 
		The number of runs in the protein quants matrix
	n_prot_factors : int, optional,
		The number of factors to embed each protein with
	n_run_factors : int, optional,
		The number of factors to use for the matrix factorization-
		based run embeddings
	n_layers : int, optional, 
		The number of hidden layers in the DNN.
	n_nodes : int, optional,
		The number of nodes in the factorization based neural 
		network. 
	rand_seed : int, optional,
		The random seed. Default is None
	testing : bool, optional. 
		Default is "False". Is the model being run in testing mode?
	biased : bool, optional,
		Use the biased mini-batch selection procedure when creating
		the data loader? 
	device : str, optional,
		The device to use for computation. {"cpu", "cuda"}
	"""
	def __init__(
		self, 
		n_prots,
		n_runs, 
		n_prot_factors=128,
		n_run_factors=128,
		n_layers=2,
		n_nodes=128,
		rand_seed=None,
		testing=False,
		biased=True,
		device="cpu",
	):
		super().__init__(
			n_prots=n_prots,
			n_runs=n_runs, 
			n_prot_factors=n_prot_factors,
			n_run_factors=n_run_factors,
			n_layers=n_layers,
			n_nodes=n_nodes,
			rand_seed=rand_seed,
			testing=testing,
			biased=biased,
			device=device,
		)

	def forward(self, locs):
		"""
		The forward pass of the model. For the current mini-batch,
		get the corresponding protein and run factors, concat
		and feed through the neural network to get a prediction. 

		Parameters
		----------
		locs : torch.tensor, of shape (batch_size, 2)
			The matrix indicies corresponding to the current
			mini-batch

		Returns
		----------
		torch.tensor of shape (batch_size, )
			The predicted values for the specified indices
		"""
		# Get the protein factors corresponding to the current batch
		prot_emb = self.prot_factors[locs[:,0],:]

		# Grab the 1th indicies of locs. Use these
		#    to index the cols of `self.run_factors`
		col_factors = self.run_factors[:,locs[:,1]].T

		# Concat and feed to the NN
		factors = torch.cat([prot_emb, col_factors], axis=1)

		preds = self.dnn(factors)
		preds = preds.squeeze(dim=1)
		
		return preds

@click.command()
@click.argument("csv", required=True, nargs=1)

@click.option("--n_prot_factors", default=128, 
    help="Number of protein factors", required=False, type=int)
@click.option("--n_run_factors", default=128, 
    help="Number of run factors", required=False, type=int)
@click.option("--n_layers", default=2, 
    help="Number of hidden layers", required=False, type=int)
@click.option("--n_nodes", default=1024, 
    help="Number of nodes per layer", required=False, type=int)
@click.option("--rand_seed", default=None, help="Random seed",
    required=False, type=int)
@click.option("--biased", default=True, 
    help="Biased batch selection?", required=False, type=bool)
@click.option("--device", default="cpu", 
    help="The device to load model on", required=False, type=str)
@click.option("--mode", default="run", 
    help="The model run mode.", required=False, type=str)

def impute(
        csv, 
        n_prot_factors, 
        n_run_factors, 
        n_layers, 
        n_nodes, 
        rand_seed, 
        biased, 
        device,
        mode, 
):
    """Impute missing values in a protein or peptide quantifications matrix."""

    # Read in the csv
    mat_pd = pd.read_csv(csv, index_col=0)
    rows = list(mat_pd.index)
    cols = list(mat_pd.columns)
    mat = np.array(mat_pd)

    test_bool = False
    if mode == "Testing":
    	test_bool = True

    Path("results/").mkdir(parents=True, exist_ok=True)

    # Init the model 
    model = Lupine(  
                n_prots=mat.shape[0],
                n_runs=mat.shape[1], 
                n_prot_factors=n_prot_factors,
                n_run_factors=n_run_factors,
                n_layers=n_layers,
                n_nodes=n_nodes,
                rand_seed=rand_seed,
                testing=test_bool,
                biased=biased,
                device=device
    )
    # Fit the model 
    print("fitting model")
    model_recon = model.fit_transform(mat)

    print("done!")
    model_recon_pd = \
    	pd.DataFrame(model_recon, index=rows, columns=cols)
    pd.DataFrame(model_recon_pd, "results/lupine_recon_quants.csv")
