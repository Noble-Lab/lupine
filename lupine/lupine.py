"""
LUPINE
8.4.24

This modules contains the `Lupine` class and the implementation of
the `impute` command. `Lupine` is the high-level implementation for
a PyTorch model for imputing protein-level quantifications using
a multilayer perceptron. Missing values are imputed by taking the
concatenation of the corresponding protein and run factors and 
feeding them through a deep neural network.

This module implements the method's `impute` command, which fits an
ensemble of Lupine models to the provided matrix and writes a single 
consensus imputed quants matrix as output. 
"""
from lupine.lupine_base import LupineBase
import torch
import click
import pandas as pd
import numpy as np
import torch
import shutil

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

@click.option("--outpath", required=True, nargs=1, type=str,
	help="Output directory")
@click.option("--n_models", default=10, 
	help="The number of models to fit.", required=False, type=int)
@click.option("--biased", default=True, 
	help="Biased batch selection?", required=False, type=bool)
@click.option("--device", default="cpu", 
	help="The device to load model on", required=False, type=str)
@click.option("--mode", default="run", 
	help="The model run mode.", required=False, type=str)

def impute(
		csv, 
		outpath,
		n_models,
		biased, 
		device,
		mode, 
):
	"""
	Impute missing values in a protein or peptide quantifications
	matrix.
	"""

	# Read in the csv
	mat_pd = pd.read_csv(csv, index_col=0)
	rows = list(mat_pd.index)
	cols = list(mat_pd.columns)
	mat = np.array(mat_pd)

	test_bool = False
	if mode == "Testing":
		test_bool = True

	# Define the full hyperparam search spaces a
	gen = np.random.default_rng(seed=18)
	n_layers_hparam_space=[1, 2]
	n_factors_hparam_space=[32, 64, 128, 256]
	n_nodes_hparam_space=[256, 512, 1024, 2048]

	print(" ")
	print("----------------------------------")
	print("--------   L U P I N E   ---------")
	print("----------------------------------")
	print(" ")
	print(f"Fitting ensemble of models on: {device}\n")

	Path(outpath).mkdir(parents=True, exist_ok=True)
	Path(outpath+"/tmp").mkdir(parents=True, exist_ok=True)

	fnames = []

	# The driver loop for ensemble model
	for n_iter in range(0, n_models): 
		print(f"Fitting model {n_iter+1} of {n_models}")

		# Randomly select the hparams
		n_layers_curr = gen.choice(n_layers_hparam_space)
		prot_factors_curr = gen.choice(n_factors_hparam_space)
		run_factors_curr = gen.choice(n_factors_hparam_space)
		n_nodes_curr = gen.choice(n_nodes_hparam_space)

		curr_seed = gen.integers(low=1, high=1e4)

		# Init an individual model 
		model = Lupine(  
					n_prots=mat.shape[0],
					n_runs=mat.shape[1], 
					n_prot_factors=prot_factors_curr,
					n_run_factors=run_factors_curr,
					n_layers=n_layers_curr,
					n_nodes=n_nodes_curr,
					rand_seed=curr_seed,
					testing=test_bool,
					biased=biased,
					device=device
		)

		# Fit the individual model 
		model_recon = model.fit_transform(mat)
		model_recon_pd = \
			pd.DataFrame(model_recon, index=rows, columns=cols)

		# Write. 
		#   These filenames may be helpful for debugging. 
		outpath_curr = \
			outpath + "tmp/lupine_imputed_" + \
			str(n_layers_curr) + "layers_" + \
			str(prot_factors_curr) + "protFactors_" + \
			str(run_factors_curr) + "runFactors_" + \
			str(n_nodes_curr) + "nodes_" + \
			str(curr_seed) + "seed" + ".csv"

		fnames.append(outpath_curr)
		model_recon_pd.to_csv(outpath_curr)

	# Do the model ensembling
	qmats = []
	for fname in fnames:
		tmp = pd.read_csv(fname, index_col=0)
		qmats.append(tmp)

	qmats_mean = np.mean(qmats, axis=0)
	outpath_ensemble = outpath + "lupine_recon_quants.csv"
	pd.DataFrame(qmats_mean, index=rows, columns=cols).\
		to_csv(outpath_ensemble)
	shutil.rmtree(outpath+"tmp")

	print(" ")
	print("Done!")
	print("----------------------------------")
	print("----------------------------------")
	print(" ")
