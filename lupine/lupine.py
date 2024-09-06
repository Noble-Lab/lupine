"""
LUPINE

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
import click
import os 
import sys
import pandas as pd
import numpy as np
import torch
import shutil
from Bio import SeqIO

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

@click.command()
@click.option("--csv", required=True, nargs=1, type=str,
	help="Path to the CSV file containing the MS runs to impute")
@click.option("--log_transform", required=True, nargs=1, type=bool,
	help="Log transform the MS runs? {True, False}")

def join(csv, log_transform):
	"""
	Add your MS runs to Lupine's training matrix, prior to 
	Lupine imputation. 
	"""
	print("\npreparing to join the MS runs...")

	# Unzip the joint quants matrix
	if os.path.isfile("data.zip"):
		cmd = "unzip data.zip"
		#os.system(cmd)
		#os.remove("data.zip")

	if not os.path.isdir("data"):
		sys.exit("Please navigate to the lupine package directory")

	# Read in the joint quants matrix and the user's csv
	joint_mat = pd.read_csv("data/joint_quantifications.csv", 
						index_col=0)
	user_mat = pd.read_csv(csv, index_col=0)

	user_quants = np.array(user_mat)
	user_quants[user_quants == 0] = np.nan

	# The optional log transform of the user's runs
	if log_transform: 
		print("log transforming...")
		user_quants = np.log2(user_quants)

	# Get the means and stds of the user's runs
	q_mean = np.nanmean(user_quants)
	q_std = np.nanstd(user_quants)
	q_var = np.nanvar(user_quants)

	# Normalize
	print("normalizing...\n")
	user_quants_norm = (user_quants - q_mean) / q_std

	# Do the right shift
	q_min = np.nanmin(user_quants_norm)
	user_quants_norm = user_quants_norm + np.abs(q_min)

	user_mat_norm = pd.DataFrame(
						user_quants_norm, 
						index=user_mat.index, 
						columns=user_mat.columns,
	)


	inter = list(np.intersect1d(user_mat_norm.index, joint_mat.index))
	joint_unique = set(joint_mat.index) - set(user_mat_norm.index)
	user_unique = set(user_mat_norm.index) - set(joint_mat.index)

	print(f"num shared proteins: {len(inter)}")
	print(f"num unique proteins: {len(user_unique)}\n")

	# Add the unique proteins to the joint quants matrix
	print("merging...")
	for ensg in user_unique:
		joint_mat.loc[ensg] = np.nan

	# Reindex the user's matrix 
	user_mat_norm = user_mat_norm.reindex(joint_mat.index)

	# Do the merge
	joint_merge = joint_mat.merge(
						user_mat_norm, 
						left_on=joint_mat.index, 
						right_on=user_mat_norm.index,
	)
	print(f"merged matrix shape: {joint_merge.shape}")

	# Reindex
	joint_merge.index = list(joint_merge["key_0"])
	joint_merge = joint_merge.drop(columns=["key_0"])

	# Write
	print("writing...")
	joint_merge.to_csv("data/joint_merged.csv")

	print("done!\n")

@click.command()
@click.option("--csv", required=True, nargs=1, type=str,
	help="Path to the CSV file containing the MS runs to impute")
@click.option("--prot_format", required=True, nargs=1, type=str,
	help="The current protein ID format. {ENSG, HGNC}")

def convert(csv, prot_format):
	"""
	Convert between ENSG or HGNC protein identifiers to ENSPs. 
	"""
	# Unzip the joint quants matrix
	if os.path.isfile("data.zip"):
		cmd = "unzip data.zip"
		#os.system(cmd)
		#os.remove("data.zip")

	if not os.path.isdir("data"):
		sys.exit("Please navigate to the lupine package directory")

	user_mat = pd.read_csv(csv, index_col=0)

	# Read in the HGNC database file
	hgnc_db = pd.read_csv(
				"data/HGNC_database.txt", 
				sep="\t", 
				low_memory=False,
	)
	# Read in the ENSEMBL fasta
	fasta_seqs = SeqIO.parse(
					open("data/gencode.v44.pc_translations.fa"), 
					"fasta",
	)

	# Create a dictionary mapping ENSPs to ENSGs
	#   And vice versa
	gene_x_prot = {}
	prot_x_gene = {}

	for fasta in fasta_seqs:
		name, descript, sequence = \
			fasta.id, fasta.description, str(fasta.seq)
		# Get the ENSP and ENSG IDs
		ensp_id = name.split("|")[0]
		ensg_id = name.split("|")[2]
		# Strip the ".x" characters. Hope this is ok.
		ensp_id = ensp_id.split(".")[0]
		ensg_id = ensg_id.split(".")[0]

		# Update the first dictionary
		prot_x_gene[ensp_id] = ensg_id

		# Update the second
		if ensg_id in gene_x_prot:
			gene_x_prot[ensg_id].append(ensp_id)
		else:
			gene_x_prot[ensg_id] = [ensp_id]

	# Init a dataframe that has all three identifiers for every
	#   protein in the user's matrix
	id_mapper = pd.DataFrame(columns=["HGNC", "ENSP", "ENSG"])

	if prot_format == "HGNC":
		id_mapper["HGNC"] = user_mat.index
	elif prot_format == "ENSG":
		id_mapper["ENSG"] = user_mat.index

	# Use the HGNC database to fill in the ENSGs
	if prot_format == "HGNC":
		for i in range(0, id_mapper.shape[0]):
			curr_row = id_mapper.iloc[i]
			curr_hgnc = curr_row["HGNC"]

			hgnc_db_row = hgnc_db[hgnc_db["symbol"] == curr_hgnc]
			curr_ensg = hgnc_db_row["ensembl_gene_id"]

			try: 
				id_mapper.loc[i, "ENSG"] = curr_ensg.item()
			except ValueError:
				id_mapper.loc[i, "ENSG"] = None

	# Use the dictionary to fill in the ENSPs
	#   This gets complicated when you have multiple ENSPs to each 
	#   ENSG. Here we're just picking the first ENSP for each ENSG. 
	#   This probably isn't ideal. 
	for i in range(0, id_mapper.shape[0]):
		curr_row = id_mapper.iloc[i]
		curr_ensg = curr_row["ENSG"]

		try: 
			curr_ensp = gene_x_prot[curr_ensg][0]
		except KeyError:
			curr_ensp = None

		id_mapper.loc[i, "ENSP"] = curr_ensp

	# Reindex the user's quants matrix
	user_mat.index = list(id_mapper["ENSP"])

	# Join by shared protein IDs
	user_mat = user_mat.groupby(by=user_mat.index).mean()

	# Write
	user_mat.to_csv("data/joint_quants_converted.csv")
