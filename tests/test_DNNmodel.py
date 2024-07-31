"""
TEST_DNNMODEL

Unit tests for the DNNImputer class. Again, some of these
may be closer to integration tests than unit tests. Note that 
because this is a DNN model, and because DNNs have a lot of params
and require a lot of data to train, we cannot approach testing 
this model like we did the standard NMF model. We shouldn't 
expect this model to converge on high-quality factorizations of
small test matrices, for example. 
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
from nmf_models import NNImputer
from utils import *

class NNImputerTesterBasic(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		"""
		A class object to test the NNImputer class. Here we're 
		generating a (12,4) matrix of random values, initializing
		an NN model and fitting the model to the dataset. Note that
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
		self.n_nodes = 8
		lr = 0.01
		patience = 10

		# model evaluation params
		self.pearson_train_thres = 0.9
		self.pearson_val_thres = 0.7

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

		# init the NN imputer model
		self.nn_model = NNImputer(
							n_rows=self.train.shape[0], 
							n_cols=self.train.shape[1], 
							n_row_factors=self.n_factors, 
							n_col_factors=self.n_factors, 
							n_nodes=self.n_nodes,
							batch_size=self.batch_size, 
							n_epochs=self.max_epochs, 
							stopping_tol=self.tolerance, 
							learning_rate=lr,
							patience=patience,
							testing=True,
							seed=42,
		)
		# save the initial latent factor weights
		self.W_orig = np.array(self.nn_model.W.data.detach())
		self.H_orig = np.array(self.nn_model.H.data.detach())

		# save the initial DNN weights -- first and second layers
		self.dnn_w_l1 = np.array(self.nn_model.dnn[0].weight.detach())
		self.dnn_w_l2 = np.array(self.nn_model.dnn[2].weight.detach())

		# fit & transform
		self.recon = \
				self.nn_model.fit_transform(self.train, self.val)

		# save the final, post-training latent factor weights
		self.W_trained = np.array(self.nn_model.W.data.detach())
		self.H_trained = np.array(self.nn_model.H.data.detach())

		# save the final DNN weights -- first and second layers
		self.dnn_w_l1_f = np.array(self.nn_model.dnn[0].weight.detach())
		self.dnn_w_l2_f = np.array(self.nn_model.dnn[2].weight.detach())

	def test_error_decrease(self):
		"""
		Are training and validation error actually decreasing? 
		"""
		iter1_t_err = np.array(self.nn_model.history["Train"])[1] 
		iterf_t_err = np.array(self.nn_model.history["Train"])[-1]

		iter1_v_err = np.array(
						self.nn_model.history["Validation"])[1] 
		iterf_v_err = np.array(
						self.nn_model.history["Validation"])[-1]

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
	
	def test_factor_matrix_shapes(self):
		"""
		Are the latent factor matrices---W and H---the expected 
		shapes?
		"""
		assert self.nn_model.W.data.shape == \
							(self.n_rows, self.n_factors)
		assert self.nn_model.H.data.shape == \
							(self.n_factors, self.n_cols)

	def test_nn_shapes(self):
		"""
		Are the layers of the model's DNN the expected shapes? 
		"""
		assert self.nn_model.dnn[0].weight.shape \
								== (2*(self.n_factors),self.n_nodes)
		assert self.nn_model.dnn[2].weight.shape == (1,self.n_nodes)

	def test_learning_nn_weights(self):
		"""
		Is the model actually learning the NN params? In other 
		words, are the NN weights actually changing as we train?
		"""
		dnn_w_l1 = pd.DataFrame(np.float32(self.dnn_w_l1)) 
		dnn_w_l2 = pd.DataFrame(np.float32(self.dnn_w_l2)) 

		dnn_w_l1_f = pd.DataFrame(np.float32(self.dnn_w_l1_f)) 
		dnn_w_l2_f = pd.DataFrame(np.float32(self.dnn_w_l2_f))

		assert not dnn_w_l1.equals(dnn_w_l1_f) 
		assert not dnn_w_l2.equals(dnn_w_l2_f) 

class NNImputerQuantsMat(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		"""
		Testing the `NNImputer` class on a real live peptide quants
		matrix. The issue is that any test that involves training 
		the full NN model will take a long time, even for this small
		test dataset. 
		"""
		# partitioning params
		quantile_anchor=0.3
		part_std=0.6
		b_prob=0.65
		min_present=4
		valid_fraction=0.1
		test_fraction=0.1
		seed=18

		# model training params
		self.batch_size=128
		self.max_epochs=200
		self.tolerance=0.01
		self.nodes=8
		self.learning_rate=0.01
		self.patience=10

		# read in the peptide quants test file
		TESTFILE = "peptide_quants_tester.csv"
		test_file = pd.read_csv(TESTFILE)
		quants = np.array(test_file)

		# MNAR partition
		self.train_mnar, self.val_mnar, test_mnar = \
			mnar_partition_thresholds_matrix(
				quants,
				q_anchor=quantile_anchor,
				t_std=part_std, 
				brnl_prob=b_prob,
				min_pres=min_present, 
				rand_state=seed,
		)
		# MCAR partition 
		self.train_mcar, self.val_mcar, test_mcar = \
			mcar_partition(
				quants, 
				val_frac=valid_fraction, 
				test_frac=test_fraction, 
				min_present=min_present, 
				random_state=seed,
		)

	def test_uneven_factors(self):
		"""
		Make sure the model can handle the case where we give it
		an uneven number of row and column latent factors
		"""
		# init the NN imputer model
		nn_model1 = NNImputer(
					n_rows=self.train_mnar.shape[0], 
					n_cols=self.train_mnar.shape[1], 
					n_row_factors=2, 
					n_col_factors=8, 
					n_nodes=self.nodes,
					batch_size=self.batch_size, 
					n_epochs=self.max_epochs, 
					stopping_tol=self.tolerance, 
					learning_rate=self.learning_rate,
					patience=self.patience,
					testing=False,
					seed=42,
		)
		# fit & transform
		recon1 = \
			nn_model1.fit_transform(self.train_mnar, self.val_mnar)

		assert recon1.shape == self.train_mnar.shape
		assert nn_model1.W.shape == (self.train_mnar.shape[0], 2)
		assert nn_model1.H.shape == (8, self.train_mnar.shape[1])

		# repeat, where the row and col factors have been reversed
		nn_model2 = NNImputer(
					n_rows=self.train_mnar.shape[0], 
					n_cols=self.train_mnar.shape[1], 
					n_row_factors=8, 
					n_col_factors=2, 
					n_nodes=self.nodes,
					batch_size=self.batch_size, 
					n_epochs=self.max_epochs, 
					stopping_tol=self.tolerance, 
					learning_rate=self.learning_rate,
					patience=self.patience,
					testing=False,
					seed=42,
		)
		# fit & transform
		recon2 = \
			nn_model2.fit_transform(self.train_mnar, self.val_mnar)

		assert recon2.shape == self.train_mnar.shape
		assert nn_model2.W.shape == (self.train_mnar.shape[0], 8)
		assert nn_model2.H.shape == (2, self.train_mnar.shape[1])

		# assert that training and validation error are decreasing
		iter1_t_err = np.array(nn_model2.history["Train"])[1] 
		iterf_t_err = np.array(nn_model2.history["Train"])[-1]

		iter1_v_err = np.array(
						nn_model2.history["Validation"])[1] 
		iterf_v_err = np.array(
						nn_model2.history["Validation"])[-1]

		assert iter1_t_err > iterf_t_err
		assert iter1_v_err > iterf_v_err

	def test_learning_weights_quants(self):
		"""
		Make sure the latent factor weights are in fact being learned. 
		Similar to an earlier test except that here we're training our
		model on real live peptide quants. 
		"""
		nn_model = NNImputer(
					n_rows=self.train_mnar.shape[0], 
					n_cols=self.train_mnar.shape[1], 
					n_row_factors=4, 
					n_col_factors=4, 
					n_nodes=self.nodes,
					batch_size=self.batch_size, 
					n_epochs=self.max_epochs, 
					stopping_tol=self.tolerance, 
					learning_rate=self.learning_rate,
					patience=self.patience,
					testing=False,
					seed=42,
		)
		# save the initial latent factor weights
		W_orig = np.array(nn_model.W.data.detach())
		H_orig = np.array(nn_model.H.data.detach())

		# fit & transform
		recon = \
			nn_model.fit_transform(self.train_mnar, self.val_mnar)

		# save the final, post-training latent factor weights
		W_trained = np.array(nn_model.W.data.detach())
		H_trained = np.array(nn_model.H.data.detach())

		W_orig_pd = pd.DataFrame(np.float32(W_orig))
		W_final_pd = pd.DataFrame(np.float32(W_trained))
		H_orig_pd = pd.DataFrame(np.float32(H_orig))
		H_final_pd = pd.DataFrame(np.float32(H_trained))

		assert not W_orig_pd.equals(W_final_pd)
		assert not H_orig_pd.equals(H_final_pd)
