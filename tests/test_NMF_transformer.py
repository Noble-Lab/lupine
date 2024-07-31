"""
TEST_NMF_TRANSFORMER

Test suite for the NMF + transformer imputation model. Most tests
rely on a small downsampled MaxQuant dataset that has peptide quants
and the associated sequences and charges. 
"""
import sys
import unittest
import pytest
import torch
import numpy as np
import pandas as pd

sys.path.append("../bin")

from transformer_models import *
from utils import *

class NMFTransformerTester(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		"""
		The setup class method for `NMFTransformerTester`. 
		We read in a small test matrix that has both peptide
		quants and the associated peptide sequences and charges,
		and use that for evaluating our transformer factorization
		model.
		"""
		mq_tester_path = "maxquant_tester_df.csv"

		# Partitioning params
		q_anchor=0.3
		t_std=1.0
		brnl_prob=0.75
		min_present=0
		rand_seed=18

		# NMF model training configs
		n_pep_facts=16
		lr=0.01
		batch_size=128
		epochs=16
		tol=0.001
		seed=18
		attn_heads=8
		ff_nodes=128
		dropout_prob=0.2
		ff_layers=1
		non_negative=True
		dnn_bool=False
		patience=10

		# The amino acids dictionary
		canonical = {
			"G": 57.021463735,
			"A": 71.037113805,
			"S": 87.032028435,
			"P": 97.052763875,
			"V": 99.068413945,
			"T": 101.047678505,
			"C": 103.009184505,
			"L": 113.084064015,
			"I": 113.084064015,
			"N": 114.042927470,
			"D": 115.026943065,
			"Q": 128.058577540,
			"K": 128.094963050,
			"E": 129.042593135,
			"M": 131.040484645,
			"H": 137.058911875,
			"F": 147.068413945,
			"U": 150.953633405,
			"R": 156.101111050,
			"Y": 163.063328575,
			"W": 186.079312980,
			"O": 237.147726925,
		}
		# load the data
		tester = pd.read_csv(mq_tester_path)
		quants = np.array(tester.iloc[:, :24])
		seqs = list(tester["seqs"])
		charges = torch.tensor(tester["charges"])

		# MNAR partition 
		train, val, test = \
				mnar_partition_thresholds_matrix(
									quants,
									q_anchor=q_anchor,
									t_std=t_std,
									brnl_prob=brnl_prob,
									min_pres=min_present,
									rand_state=rand_seed,
		)
		# init the NMF transformer model 
		self.nmf_trans_model = TransformerFactorizationImputer(  
							n_runs=train.shape[1], 
							aa_dict=canonical, 
							seqs=seqs, 
							charges=charges,
							n_peptide_factors=n_pep_facts, 
							n_run_factors=n_pep_facts,
							learning_rate=lr,
							batch_size=batch_size,
							tolerance=tol,
							max_epochs=epochs,
							patience=patience,
							attn_heads=attn_heads,
							feedforward_nodes=ff_nodes,
							trans_layers=ff_layers,
							dropout_prob=dropout_prob,
							rand_seed=seed,
							dnn=False,
							non_negative=True,
							testing=False,
							biased=True,
							device="cpu",
		)
		# record the original (untrained) run factors
		self.run_factors_orig = \
			np.array(self.nmf_trans_model.run_factors.data.detach())
		# and the original amino acid embeddings
		self.aa_embed_orig = \
			np.array(self.nmf_trans_model.encoder.aa_encoder.\
										weight.data.detach())

		# fit and transform the model 
		self.recon = \
			self.nmf_trans_model.fit_transform(train, val)

	def test_model_construction(self):
		"""
		A very basic test to ensure that the 
		`TransformerFactorizationImputer` model is being constructed
		in the way that it should be.
		"""
		# some basic checks for model params
		params = [
			"run_factors", 
			"encoder.charge_encoder.weight",
			"encoder.aa_encoder.weight", 
			"encoder.transformer_encoder.layers.0.linear1.bias",
			"encoder.transformer_encoder.layers.0.linear1.weight",
		]
		for param in params:
			assert param in self.nmf_trans_model.state_dict().keys()

		assert len(self.nmf_trans_model.state_dict().keys()) == 17
		# 16 is the number of embedding dims
		assert self.nmf_trans_model.encoder.aa_encoder.embedding_dim\
																== 16
		# 24 is the number of runs in the training matrix
		assert self.nmf_trans_model.encoder.aa_encoder.num_embeddings\
																 == 24
		# check for the number of model params
		params = self.nmf_trans_model.parameters()
		n_params = 0
		for x in params:
			n_params += 1
		assert n_params == 15

		# check for the shape of the model's run_factors
		assert self.nmf_trans_model.run_factors.shape == (16,24)

		# check for proper construction of the amino acids dict
		assert len(self.nmf_trans_model.aa_dict.keys()) == 22

	def test_device_type(self):
		"""
		Make sure that all of the model's params are registered on 
		the specified device, which is CPU in this case. This is 
		challenging to do in the GPU case because my work station
		doesn't have a GPU. 
		"""
		assert self.nmf_trans_model.device == "cpu"
		assert self.nmf_trans_model.run_factors.device.type == "cpu"
		assert self.nmf_trans_model.encoder.device.type == "cpu"
		assert self.nmf_trans_model.charges.device.type == "cpu"

	def test_model_recon(self):
		"""
		Just test the the `TransformerFactorizationImputer`
		model `fit_transform` method executes without error
		and successfully transforms the training matrix.
		"""
		# make sure the reconstruction has no NaNs
		assert np.count_nonzero(np.isnan(self.recon)) == 0
		assert self.recon.shape == (64, 24)

		# make sure that the model trained for all 16 epochs
		assert list(self.nmf_trans_model.history["epoch"])[-1] \
															== 16

	def test_learning_embeddings(self):
		"""
		Make sure that the model is training the run factors and
		the amino acid embeddings. These should both be parameters
		of the model. 
		"""
		# record the final (trained) run factors
		run_factors_final = \
			np.array(self.nmf_trans_model.run_factors.data.detach())

		# run check
		run_factors_orig = pd.DataFrame(self.run_factors_orig)
		run_factors_final = pd.DataFrame(run_factors_final)
		assert not run_factors_orig.equals(run_factors_final)

		# do the same thing for the final (trained) AA embeddings
		aa_embed_final = \
			np.array(self.nmf_trans_model.encoder.aa_encoder.\
										weight.data.detach())

		# check
		aa_embed_orig = pd.DataFrame(self.aa_embed_orig)
		aa_embed_final = pd.DataFrame(aa_embed_final)
		assert not aa_embed_orig.equals(aa_embed_final)

	def test_nonnegativity_constraint_nmf(self):
		"""
		Make sure that the model's run_factors were indeed 
		constrained to be greater than zero
		"""
		assert torch.min(self.nmf_trans_model.run_factors.data) >= 0.0

	def test_decreasing_loss_nmf(self):
		"""
		Test that the loss is decreasing as the model trains. 
		Here we're comparing the loss from the first four training
		epochs to the last, for both training and validation sets.
		"""
		# For the validation set
		valid_start = \
			list(self.nmf_trans_model.history["Validation"])[:4]
		valid_end = \
			list(self.nmf_trans_model.history["Validation"])[-4:]

		valid_start_mean = np.mean(valid_start)
		valid_end_mean = np.mean(valid_end)

		assert valid_start_mean > valid_end_mean

		# For the training set
		train_start = \
			self.nmf_trans_model.history["Train"][:4]
		train_end = \
			self.nmf_trans_model.history["Train"][-4:]

		train_start_mean = np.mean(train_start)
		train_end_mean = np.mean(train_end)

		assert train_start_mean > train_end_mean

class NMFTransformerConvergenceTester(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		"""
		The setup class method for `NMFTransformerConvergenceTester`. 
		Reading in the small maxquants tester dataframe again. This 
		time we want to bump up the `max_iters` and make sure the 
		model will successfully train until convergence. 
		"""
		mq_tester_path = "maxquant_tester_df.csv"

		# Partitioning params
		q_anchor=0.3
		t_std=1.0
		brnl_prob=0.75
		min_present=0
		rand_seed=18

		# NN model training configs
		n_pep_facts=16
		lr=0.01
		batch_size=128
		self.epochs=512
		tol=0.001
		seed=42
		attn_heads=8
		ff_nodes=128
		dropout_prob=0.2
		ff_layers=1
		non_negative=True
		dnn_bool=False
		patience=10

		# The amino acids dictionary
		self.canonical = {
			"G": 57.021463735,
			"A": 71.037113805,
			"S": 87.032028435,
			"P": 97.052763875,
			"V": 99.068413945,
			"T": 101.047678505,
			"C": 103.009184505,
			"L": 113.084064015,
			"I": 113.084064015,
			"N": 114.042927470,
			"D": 115.026943065,
			"Q": 128.058577540,
			"K": 128.094963050,
			"E": 129.042593135,
			"M": 131.040484645,
			"H": 137.058911875,
			"F": 147.068413945,
			"U": 150.953633405,
			"R": 156.101111050,
			"Y": 163.063328575,
			"W": 186.079312980,
			"O": 237.147726925,
		}
		# load the data
		tester = pd.read_csv(mq_tester_path)
		quants = np.array(tester.iloc[:, :24])
		self.seqs = list(tester["seqs"])
		self.charges = torch.tensor(tester["charges"])

		# MNAR partition 
		self.train, self.val, test = \
				mnar_partition_thresholds_matrix(
									quants,
									q_anchor=q_anchor,
									t_std=t_std,
									brnl_prob=brnl_prob,
									min_pres=min_present,
									rand_state=rand_seed,
		)
		# init the NMF transformer model 
		self.nmf_model = TransformerFactorizationImputer(  
							n_runs=self.train.shape[1], 
							aa_dict=self.canonical, 
							seqs=self.seqs, 
							charges=self.charges,
							n_peptide_factors=n_pep_facts, 
							n_run_factors=n_pep_facts,
							learning_rate=lr,
							batch_size=batch_size,
							tolerance=tol,
							max_epochs=self.epochs,
							patience=patience,
							attn_heads=attn_heads,
							feedforward_nodes=ff_nodes,
							trans_layers=ff_layers,
							dropout_prob=dropout_prob,
							rand_seed=seed,
							dnn=False,
							non_negative=True,
							testing=False,
							biased=True,
							device="cpu",
		)
		# fit and transform the model 
		self.nmf_recon = \
			self.nmf_model.fit_transform(self.train, self.val)

	def test_nmf_convergence(self):
		"""
		Make sure that the NN transformer model successfully trained
		until completion. 
		"""
		assert list(self.nmf_model.history["epoch"])[-1] <= self.epochs
		assert self.nmf_model.stopping_criteria == "wilcoxon"

		assert list(self.nmf_model.history["Train"])[-1] < 2.0
		assert list(self.nmf_model.history["Validation"])[-1] < 35

	def test_decreasing_loss_nmf_extended(self):
		"""
		Make sure that the training and validation losses are 
		decresing as training proceeds. 
		"""
		# For the validation set
		valid_start = \
			list(self.nmf_model.history["Validation"])[:8]
		valid_end = \
			list(self.nmf_model.history["Validation"])[-8:]

		valid_start_mean = np.mean(valid_start)
		valid_end_mean = np.mean(valid_end)

		assert valid_start_mean > valid_end_mean

		# For the training set
		train_start = \
			self.nmf_model.history["Train"][:8]
		train_end = \
			self.nmf_model.history["Train"][-8:]

		train_start_mean = np.mean(train_start)
		train_end_mean = np.mean(train_end)

		assert train_start_mean > train_end_mean
