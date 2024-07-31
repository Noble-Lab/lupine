"""
TEST_NMFMODEL

Unit tests for the NMFImputer and BaseImputer classes. One could
argue that some of these are actually integration tests, as full
functionality of the NMFImputer class depends on full functionality
of a whole bunch of lower level pieces. 
"""
import sys
import unittest
import pytest
import torch
from scipy.stats import ranksums
import numpy as np
import pandas as pd
import os

sys.path.append("../bin")

from base import BaseImputer, FactorizationDataset
from nmf_models import NMFImputer
from utils import *

class NMFImputerTesterBasic(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		"""
		A class object to test the NMFImputer class. Here we're 
		generating a (12,4) matrix of random values, initializing
		an NMF model and fitting the model to the dataset. Note that
		values in this matrix are between 0 and 1. 
		"""
		# data generation params
		rand_seed = 18
		self.n_cols = 4
		self.n_rows = 12

		# standard model training params
		self.batch_size = 128
		self.max_epochs = 200
		self.n_factors = 4
		self.tolerance = 1e-3
		self.lr = 0.01
		self.patience = 10

		# model evaluation params
		self.pearson_train_thres = 0.9
		self.pearson_val_thres = 0.3

		# init numpy's random number generator
		rng = np.random.default_rng(seed=rand_seed)

		W = np.matrix(rng.random(self.n_rows))
		H = np.matrix(rng.random(self.n_cols))
		X = np.matmul(W.T, H)

		# training set has six entries missing
		self.train = X.copy()
		self.train[2,1] = np.nan
		self.train[0,0] = np.nan
		self.train[7,1] = np.nan
		self.train[8,0] = np.nan
		self.train[10,3] = np.nan
		self.train[6,3] = np.nan

		# validation set has only six entries
		self.val = np.zeros(X.shape)

		self.val[2,1] = X[2,1]
		self.val[0,0] = X[0,0]
		self.val[7,1] = X[7,1]
		self.val[8,0] = X[8,0]
		self.val[10,3] = X[10,3]
		self.val[6,3] = X[6,3]

		self.val[self.val == 0.0] = np.nan

		# init the NMF imputer model
		self.nmf_model = NMFImputer(
							n_rows=self.train.shape[0], 
							n_cols=self.train.shape[1], 
							n_row_factors=self.n_factors, 
							n_col_factors=self.n_factors, 
							batch_size=self.batch_size, 
							n_epochs=self.max_epochs, 
							stopping_tol=self.tolerance, 
							learning_rate=self.lr,
							patience=self.patience,
							testing=True,
							log_transform=False,
							seed=42,
		)
		# save the initial latent factor weights
		self.W_orig = np.array(self.nmf_model.W.data.detach())
		self.H_orig = np.array(self.nmf_model.H.data.detach())

		# fit & transform
		self.recon = self.nmf_model.fit_transform(self.train, self.val)

		# save the trained latent factor weights
		self.W_trained = np.array(self.nmf_model.W.data.detach())
		self.H_trained = np.array(self.nmf_model.H.data.detach())

	def test_reconstruction_basic(self):
		"""
		Test that some basic attributes of the reconstructed matrix
		look ok. 
		"""
		assert self.recon.shape == self.train.shape
		assert np.min(self.recon) >= 0
		assert np.max(self.recon) < 1

		lf_recon = torch.mm(self.nmf_model.W, self.nmf_model.H)
		assert lf_recon != self.recon

	def test_reconstruction_error(self):
		"""
		Tests how well the model did at reconstructing the matrix. 
		"""
		# again, not sure how conservative to make these thresholds
		assert np.array(
					self.nmf_model.history["Train"])[-1] < 0.1
		assert np.array(
					self.nmf_model.history["Validation"])[-1] < 1.0

	def test_factors(self):
		"""
		Test that the latent factors for the trained model look ok. 
		"""
		assert self.nmf_model.W.shape == \
									(self.n_rows, self.n_factors)
		assert self.nmf_model.H.shape == \
									(self.n_factors, self.n_cols)

		assert torch.min(self.nmf_model.W.data) >= 0
		assert torch.min(self.nmf_model.H.data) >= 0

	def test_convergence(self):
		"""
		Test that model training convergence criteria worked properly.
		"""
		assert self.nmf_model.history.shape[0] <= self.max_epochs+1

		# the Wilcoxon convergence criteria
		if self.nmf_model.stopping_criteria == "wilcoxon":
			win2 = np.array(
						self.nmf_model.history["Validation"][-5:])
			win1 = np.array(
						self.nmf_model.history["Validation"][-13:-8])
			wilcox_p = ranksums(
						win2, win1, alternative="greater")[1]

			assert wilcox_p < 0.05

		# the standard convergence criteria
		elif self.nmf_model.stopping_criteria == "standard":
			val_min = np.min(
						self.nmf_model.history["Validation"])
			val_last = np.array(
						self.nmf_model.history["Validation"])[-1]
			tol = np.abs((val_min - val_last) / val_min)

			assert tol < self.tolerance

		# early stopping not triggered
		else:
			assert self.nmf_model.history.shape[0] == self.max_epochs+1

	def test_correlations(self):
		"""
		Tests how well correlated the reconstructed and ground truth
		values are, for training and validation sets. 
		"""
		def _get_pearson_r(orig_mat, recon_mat):
			"""
			Parameters
			----------
			orig_mat : np.ndarray, 
				The original matrix. {training, validation}
			recon_mat : np.ndarray, 
				The reconstructed matrix

			Returns
			----------
			pearson_r : float, 
				The Pearson correlation coefficient
			"""
			orig_nans = np.isnan(orig_mat)

			# get the non-nan values for the orig and recon mats
			orig_present = orig_mat[~orig_nans]
			recon_present = recon_mat[~orig_nans]

			# get the Pearson's correlation coefficient
			corr_mat = np.corrcoef(orig_present, recon_present)
			pearson_r = np.around(corr_mat[0][1], 3)

			return pearson_r

		pearson_train = _get_pearson_r(self.train, self.recon)
		pearson_val = _get_pearson_r(self.val, self.recon)

		# not totally sure how to set these. Sometimes fails on val
		assert pearson_train > self.pearson_train_thres
		assert pearson_val > self.pearson_val_thres

	def test_forward_method(self):
		"""
		I have this old and slow code for doing full matrix 
		multiplication for every forward pass, then isolating
		just the matrix entries associated with the current mini
		batch and returning them. Its slow, but I know it works. 
		Here I want to verify that the new, faster, model forward
		pass code yields the same results as the older code. 
		"""
		# init the loader
		train_loader = FactorizationDataset(
								self.train, 
								self.val, 
								partition="Train",
								biased=False,
								batch_size=self.batch_size,
		)
		# get the locs
		train_loader.get_standard_loader()
		# run the test
		for locs, targets in train_loader:
			# the old way
			X_hat = torch.mm(self.nmf_model.W, self.nmf_model.H)
			X_hat_b = X_hat[tuple(locs.T)]
			# the current way
			preds = self.nmf_model(locs)
			assert torch.eq(X_hat_b, preds).all()

	def test_error_decrease(self):
		"""
		Make sure that the model's train and validation error
		are actually decreasing. 

		Is there anything else we could add here? 
		"""
		iter1_t_err = np.array(self.nmf_model.history["Train"])[1] 
		iterf_t_err = np.array(self.nmf_model.history["Train"])[-1]

		iter1_v_err = np.array(
						self.nmf_model.history["Validation"])[1] 
		iterf_v_err = np.array(
						self.nmf_model.history["Validation"])[-1]

		assert iter1_t_err > iterf_t_err
		assert iter1_v_err > iterf_v_err

	def test_learning_factors(self):
		"""
		Is the model actually learning the latent factors? In other
		words, are the latent factor weights actually changing as
		we train the model? 
		"""
		W_orig_pd = pd.DataFrame(np.float32(self.W_orig))
		W_final_pd = pd.DataFrame(np.float32(self.W_trained))
		H_orig_pd = pd.DataFrame(np.float32(self.H_orig))
		H_final_pd = pd.DataFrame(np.float32(self.H_trained))

		assert not W_orig_pd.equals(W_final_pd)
		assert not H_orig_pd.equals(H_final_pd)

		assert self.W_orig.shape == self.W_trained.shape
		assert self.H_orig.shape == self.H_trained.shape

	def test_factor_shapes(self):
		"""
		Are the latent factor matrices the expected shapes?
		"""
		assert self.nmf_model.W.data.shape == \
							(self.n_rows, self.n_factors)
		assert self.nmf_model.H.data.shape == \
							(self.n_factors, self.n_cols)

class NMFImputerTesterLargeRange(unittest.TestCase):
	@classmethod 
	def setUpClass(self):
		"""
		A class object for testing the NMFImputer class with a matrix
		of simulated values pulled from a much wider dynamic range. 
		The idea here is to better mimic the distribution of values
		the NMFImputer class will encounter from proteomics data. This
		is also a more realistic matrix in the sense that it has many
		more features than samples: (100,10). 
		"""
		# data generation params
		rand_seed = 36
		n_cols = 10
		n_rows = 100
		n_factors = 2

		# partitioning params
		val_frac = 0.1
		test_frac = 0.1
		min_present = 1

		# model training params
		self.batch_size = 128
		self.max_epochs = 200
		self.n_factors = 4
		self.tolerance = 1e-3
		self.lr = 0.01
		self.patience = 10

		# model evaluation params -- how to set these? 
		self.train_recon_tol = 0.5
		self.val_recon_tol = 2.0

		# init numpy's random generator
		rng = np.random.default_rng(seed=rand_seed)

		# generate random arrays for the factor matrices
		W = np.array(rng.random((n_rows, n_factors)))
		H = np.array(rng.random((n_factors, n_cols)))
		X = np.matmul(W, H)

		# blow it up
		X_bu = np.exp(np.exp(X)) * 1e5

		# partition
		self.train, self.val, self.test = mcar_partition(
						X_bu, val_frac, test_frac, min_present)

	def test_recon_inflated(self):
		"""
		Make sure that the NMFImputer model does a reasonably good
		job at reconstructing a matrix with a range that better
		resembles a proteomics experiment. 
		"""
		# init the NMF imputer model
		nmf_model = NMFImputer(
						n_rows=self.train.shape[0], 
						n_cols=self.train.shape[1], 
						n_row_factors=self.n_factors, 
						n_col_factors=self.n_factors, 
						batch_size=self.batch_size, 
						n_epochs=self.max_epochs, 
						stopping_tol=self.tolerance, 
						learning_rate=self.lr,
						patience=self.patience,
						testing=True,
						log_transform=False,
						seed=42,
		)

		# fit & transform
		recon = nmf_model.fit_transform(self.train, self.val)

		assert np.array(
					nmf_model.history["Train"])[-1] \
					< self.train_recon_tol
		assert np.array(
					nmf_model.history["Validation"])[-1] \
					< self.val_recon_tol

class NMFImputerTesterKnownFactors(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		"""
		A class object for testing the NMFImputer class with a 
		simulated matrix of known dimensionality. The NMF model 
		should be able to perfectly reconstruct the test matrix, if
		we give the model the correct number of latent factors.
		"""
		# data generation params
		rand_seed = 18
		n_cols = 10
		n_rows = 12
		n_factors = 2

		# partitioning params
		val_frac = 0.1
		test_frac = 0.1
		min_present = 0

		# model training params
		self.batch_size = 128
		self.max_epochs = 200
		self.n_factors = 4
		self.tolerance = 1e-3
		self.lr = 0.01
		self.patience = 10

		# evaluation params -- how to set these? 
		self.train_recon_tol = 0.1
		self.val_recon_tol = 0.7

		# init numpy's random generator
		rng = np.random.default_rng(seed=rand_seed)

		# generate random arrays for the factor matrices
		W = np.array(rng.random((n_rows, n_factors)))
		H = np.array(rng.random((n_factors, n_cols)))
		self.X = np.matmul(W, H)

		# partition
		self.train, self.val, self.test = mcar_partition(
						self.X, val_frac, test_frac, min_present)

	def test_recon_known_lf(self):
		"""
		How good is the model's reconstruction when I generate a 
		matrix with x latent factors, then tell the model to use
		x latent factors for training? 
		"""
		# init the NMF imputer model
		nmf_model = NMFImputer(
				        n_rows=self.train.shape[0], 
				        n_cols=self.train.shape[1], 
				        n_row_factors=self.n_factors, 
				        n_col_factors=self.n_factors, 
				        batch_size=self.batch_size, 
				        n_epochs=self.max_epochs, 
				        stopping_tol=self.tolerance, 
				        learning_rate=self.lr,
				        patience=self.patience,
				        testing=True,
				        log_transform=False,
				        seed=42,
		)

		# fit & transform
		recon = nmf_model.fit_transform(self.train, self.val)

		# Shouldn't these reconstruction errors basically be zero?
			# It seems that these tolerances should be lower than 
			# ones for the previous simulated matrix test, for ex.
		final_train_error = \
					np.array(nmf_model.history["Train"])[-1]
		final_valid_error = \
					np.array(nmf_model.history["Validation"])[-1]

		assert final_train_error < self.train_recon_tol
		assert final_valid_error < self.val_recon_tol

	# def tearDownClass():
	# 	"""
	# 	Remove the checkpointed model that the BaseImputer 
	# 	class creates.
	# 	"""
	# 	os.remove("OPT_MODEL_INTERNAL.pt")
