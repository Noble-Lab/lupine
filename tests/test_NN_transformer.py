"""
TEST_NN_TRANSFORMER

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

class NNTransformerTester(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		"""
		The setup class method for `NNTransformerTester`. 
		We read in a small test matrix that has both peptide
		quants and the associated peptide sequences and charges,
		and use that for evaluating our transformer factorization
		model. Explicitly testing the 
		`TransformerFactorizationNNImputer` model
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
		epochs=16
		tol=0.001
		seed=18
		attn_heads=8
		ff_nodes=128
		nn_nodes=16
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
		# init the NN transformer model 
		self.nn_trans_model = TransformerFactorizationNNImputer(  
							n_runs=self.train.shape[1], 
							aa_dict=self.canonical, 
							seqs=self.seqs, 
							charges=self.charges,
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
							factorization_nn_nodes=nn_nodes,
							rand_seed=seed,
							dnn=True,
							non_negative=False,
							testing=False,
							biased=True,
							device="cpu",
		)
		# record the original (untrained) run factors
		self.run_factors_orig = \
			np.array(self.nn_trans_model.run_factors.data.detach())
		# and the original amino acid embeddings
		self.aa_embed_orig = \
			np.array(self.nn_trans_model.encoder.aa_encoder.\
										weight.data.detach())
		# and the first layer of the factorization NN
		self.nn_l1_orig = \
			np.array(self.nn_trans_model.dnn[0].weight.data.detach())

		# fit and transform the model 
		self.recon = \
			self.nn_trans_model.fit_transform(self.train, self.val)

	def test_nn_model_params(self):
		"""
		Test that the NN model was properly constructed, has all of 
		the expected params, etc. 
		"""
		params = [
			"run_factors", 
			"encoder.aa_encoder.weight", 
			"dnn.0.weight", 
			"dnn.2.weight", 
			"dnn.0.bias", 
			"encoder.transformer_encoder.layers.0.linear1.weight",
		]
		for param in params:
			assert param in self.nn_trans_model.state_dict().keys()

	def test_device_type(self):
		"""
		Make sure that all of the model's params are registered on 
		the specified device, which is CPU in this case. This is 
		challenging to do in the GPU case because my work station
		doesn't have a GPU. 
		"""
		assert self.nn_trans_model.device == "cpu"
		assert self.nn_trans_model.run_factors.device.type == "cpu"
		assert self.nn_trans_model.encoder.device.type == "cpu"
		assert self.nn_trans_model.charges.device.type == "cpu"

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
		assert list(self.nn_trans_model.history["epoch"])[-1] \
															== 16

	def test_learning_embeddings(self):
		"""
		Make sure that the model is training the run factors and
		the amino acid embeddings. These should both be parameters
		of the model. 
		"""
		# record the final (trained) run factors
		run_factors_final = \
			np.array(self.nn_trans_model.run_factors.data.detach())

		# run check
		run_factors_orig = pd.DataFrame(self.run_factors_orig)
		run_factors_final = pd.DataFrame(run_factors_final)
		assert not run_factors_orig.equals(run_factors_final)

		# do the same thing for the final (trained) AA embeddings
		aa_embed_final = \
			np.array(self.nn_trans_model.encoder.aa_encoder.\
										weight.data.detach())

		# check
		aa_embed_orig = pd.DataFrame(self.aa_embed_orig)
		aa_embed_final = pd.DataFrame(aa_embed_final)
		assert not aa_embed_orig.equals(aa_embed_final)

		# and for the first layer of the factorization NN
		nn_l1_final = \
			np.array(self.nn_trans_model.dnn[0].weight.data.detach())

		nn_l1_orig = pd.DataFrame(self.nn_l1_orig)
		nn_l1_final = pd.DataFrame(nn_l1_final)
		assert not nn_l1_orig.equals(nn_l1_final)

	def test_factorization_dnn_params(self):
		"""
		Make sure that the neural network constructed for matrix 
		factorization is of the expected shape and size. 
		"""
		assert self.nn_trans_model.dnn[0].in_features == 32
		assert self.nn_trans_model.dnn[0].out_features == 16
		assert self.nn_trans_model.dnn[2].in_features == 16
		assert self.nn_trans_model.dnn[2].out_features == 1

	def test_nonnegativity_nn(self):
		"""
		The non-negativity constraint was turned off for this model, 
		so we should end up with some very negative run factors. 
		"""
		assert torch.min(self.nn_trans_model.run_factors.data) < 0.0

	def test_decreasing_loss_nn(self):
		"""
		Test that the loss is decreasing as the model trains. 
		Here we're comparing the loss from the first four training
		epochs to the last, for both training and validation sets.
		"""
		# For the validation set
		valid_start = \
			list(self.nn_trans_model.history["Validation"])[:4]
		valid_end = \
			list(self.nn_trans_model.history["Validation"])[-4:]

		valid_start_mean = np.mean(valid_start)
		valid_end_mean = np.mean(valid_end)

		assert valid_start_mean > valid_end_mean

		# For the training set
		train_start = \
			self.nn_trans_model.history["Train"][:4]
		train_end = \
			self.nn_trans_model.history["Train"][-4:]

		train_start_mean = np.mean(train_start)
		train_end_mean = np.mean(train_end)

		assert train_start_mean > train_end_mean

	def test_different_n_factors(self):
		"""
		Make sure the model still functions when I specify a 
		different number of peptide factors than run factors.
		"""
		model = TransformerFactorizationNNImputer(  
								n_runs=self.train.shape[1], 
								aa_dict=self.canonical, 
								seqs=self.seqs, 
								charges=self.charges,
								n_peptide_factors=32, 
								n_run_factors=16,
								max_epochs=16,
								feedforward_nodes=64,
								factorization_nn_nodes=8,
								dnn=True,
								non_negative=False,
		)
		# fit and transform the model 
		recon = model.fit_transform(self.train, self.val)

		# make sure the reconstruction has no NaNs
		assert np.count_nonzero(np.isnan(recon)) == 0
		assert recon.shape == (64, 24)

		# make sure that the model trained for all 16 epochs
		assert list(model.history["epoch"])[-1] == 16

class NNTransformerConvergenceTester(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		"""
		The setup class method for `NNTransformerConvergenceTester`. 
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
		seed=18
		attn_heads=8
		ff_nodes=128
		nn_nodes=16
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
		# init the NN transformer model 
		self.nn_model = TransformerFactorizationNNImputer(  
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
							factorization_nn_nodes=nn_nodes,
							rand_seed=seed,
							dnn=True,
							non_negative=False,
							testing=False,
							biased=True,
							device="cpu",
		)
		# fit and transform the model 
		self.nn_recon = \
			self.nn_model.fit_transform(self.train, self.val)

	def test_nn_convergence(self):
		"""
		Make sure that the NN transformer model successfully trained
		until completion. 
		"""
		assert list(self.nn_model.history["epoch"])[-1] <= self.epochs
		assert self.nn_model.stopping_criteria == "wilcoxon"

		assert list(self.nn_model.history["Train"])[-1] < 1.0
		assert list(self.nn_model.history["Validation"])[-1] < 2.0

	def test_decreasing_loss_nn_extended(self):
		"""
		Make sure that the training and validation losses are 
		decresing as training proceeds. 
		"""
		# For the validation set
		valid_start = \
			list(self.nn_model.history["Validation"])[:8]
		valid_end = \
			list(self.nn_model.history["Validation"])[-8:]

		valid_start_mean = np.mean(valid_start)
		valid_end_mean = np.mean(valid_end)

		assert valid_start_mean > valid_end_mean

		# For the training set
		train_start = \
			self.nn_model.history["Train"][:8]
		train_end = \
			self.nn_model.history["Train"][-8:]

		train_start_mean = np.mean(train_start)
		train_end_mean = np.mean(train_end)

		assert train_start_mean > train_end_mean
