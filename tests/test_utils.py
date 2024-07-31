"""
TEST_UTILS

Unit tests for the `utils` module. 
TODO: Test the updated MNAR partition function, which now returns
disjoint train, val & test sets. 
"""
import sys
import unittest
import pytest
import torch
import numpy as np
import pandas as pd

sys.path.append("../bin")

from utils import *

class UtilsTester(unittest.TestCase):
	@classmethod
	def setUpClass(self):
		"""
		Evaluate the methods contained within UTILS.PY. 
		"""
		# params for the simulated uniform matrix
		n_rows = 12
		n_cols = 4
		self.rand_seed = 18
		self.rng = np.random.default_rng(seed=self.rand_seed)

		# partition params
		self.val_frac = 0.1
		self.test_frac = 0.1
		self.min_present = 1
		self.q_anchor = 0.2 # these three for MNAR
		self.t_std = 0.1
		self.brnl_prob = 0.5

		# generate random arrays for the factor matrices
		W = np.matrix(self.rng.random(n_rows))
		H = np.matrix(self.rng.random(n_cols))
		self.X = np.array(np.matmul(W.T, H))

		# Gaussian matrix params
		n_rows_g = 1000
		n_cols_g = 100
		gloc = 1000
		gscale = 1000
		
		# generate the Gaussian matrix
		self.X_norm = np.abs(
						self.rng.normal(
								loc=gloc, 
								scale=gscale, 
								size=(n_rows_g,n_cols_g)
						)
		)

	def test_mcar_fracs(self):
		"""
		Test utils.mcar_partition(). Make sure that the train, 
		val and test fractions have the expected number of missing
		values.
		"""
		train, val, test = mcar_partition(
									matrix=self.X, 
									val_frac=self.val_frac, 
									test_frac=self.test_frac, 
									min_present=0,
		)
		assert train.shape == self.X.shape 
		assert val.shape == self.X.shape 
		assert test.shape == self.X.shape 

		# these should always be the same, bc I set the 
			# random seed
		assert np.count_nonzero(np.isnan(train)) == 8
		assert np.count_nonzero(np.isnan(val)) == 44

	def test_mnar_fracs(self):
		"""
		Test utils.mnar_partition_thresholds_matrix(). Make sure 
		that the train, val and test fractions have the 
		expected number of missing values.
		"""
		train, val, test = mnar_partition_thresholds_matrix(
									mat=self.X, 
									q_anchor=self.q_anchor, 
									t_std=self.t_std, 
									brnl_prob=self.brnl_prob, 
									min_pres=0,
									rand_state=self.rand_seed,
		)
		assert np.count_nonzero(np.isnan(train)) == 5
		assert np.count_nonzero(np.isnan(val)) == 46

	def test_min_present(self):
		"""
		Make sure that both partitioning methods properly handle 
		the case where I specify `min_present` > n_cols of the 
		matrix. 
		"""
		# MCAR
		train_mcar, val_mcar, test_mcar = \
								mcar_partition(
										matrix=self.X, 
										val_frac=self.val_frac, 
										test_frac=self.test_frac, 
										min_present=5,
		)

		# MNAR
		train_mnar, val_mnar, test_mnar = \
					mnar_partition_thresholds_matrix(
										mat=self.X, 
										q_anchor=self.q_anchor, 
										t_std=self.t_std, 
										brnl_prob=self.brnl_prob, 
										min_pres=5,
										rand_state=self.rand_seed,
		)

		assert not train_mcar.size > 0
		assert not val_mcar.size > 0
		assert not test_mcar.size > 0

		assert not train_mnar.size > 0
		assert not val_mnar.size > 0

	def test_mcar_distribution(self):
		"""
		Compares the means of the train and validation sets resulting
		from the MCAR partition to the original matrix. Should be
		bang on, for a large enough dataset. 
		"""
		# MCAR partition
		train_mcar, val_mcar, test_mcar = \
									mcar_partition(
												self.X_norm, 
												self.val_frac, 
												self.test_frac, 
												self.min_present,
		)
		# the mean of the original matrix
		orig_mean = np.mean(self.X_norm)

		# the MCAR set
		train_mcar_mean = np.nanmean(train_mcar)
		val_mcar_mean = np.nanmean(val_mcar)

		# make sure the MCAR means are bang on
		assert np.isclose(orig_mean, train_mcar_mean, atol=1)
		assert np.isclose(orig_mean, val_mcar_mean, atol=20)

	def test_mnar_distribution(self):
		"""
		Compares the means of the train and validation sets after
		partitioning with MNAR to the mean of the original set. The
		MNAR train set should be greater than, and the validation 
		set should be less than the original mean. 

		What else can I do here, besides comparing the means? 
		"""
		# MNAR partition
		train_mnar, val_mnar, test_mnar = \
					mnar_partition_thresholds_matrix(
										mat=self.X_norm, 
										q_anchor=self.q_anchor, 
										t_std=self.t_std, 
										brnl_prob=self.brnl_prob,
										min_pres=self.min_present,
										rand_state=self.rand_seed,
		)
		# the mean of the original matrix
		orig_mean = np.mean(self.X_norm)

		# the MNAR set
		train_mnar_mean = np.nanmean(train_mnar)
		val_mnar_mean = np.nanmean(val_mnar)

		assert train_mnar_mean > orig_mean
		assert val_mnar_mean < orig_mean

		pass

	def test_mcar_large_frac(self):
		"""
		Does the MCAR partition method respond properly when I 
		increase the specified held-out fraction? 
		"""
		# MCAR partition
		train, val, test = \
					mcar_partition(
						self.X_norm, 0.9, 0.0, self.min_present
					)

		n_train_nan = np.count_nonzero(np.isnan(train))
		n_val_nan = np.count_nonzero(np.isnan(val))
		
		assert n_train_nan == 90000
		assert n_val_nan == 10000

	def test_mnar_large_frac(self):
		"""
		Does the MNAR partition method respond properly when I 
		increase the specified held-out fraction? Generate three 
		different partitions with slightly different params, 
		assert that the number of MVs in the training set increases
		every time.

		Is there anything else I could do here?  
		"""
		train1, val1, test1 = \
					mnar_partition_thresholds_matrix(
										mat=self.X_norm, 
										q_anchor=0.2, 
										t_std=0.1, 
										brnl_prob=0.5,
										min_pres=1,
										rand_state=18,
		)
		train2, val2, test2 = \
					mnar_partition_thresholds_matrix(
										mat=self.X_norm, 
										q_anchor=0.4, 
										t_std=0.4, 
										brnl_prob=0.6,
										min_pres=1,
										rand_state=18,
		)
		train3, val3, test3 = \
					mnar_partition_thresholds_matrix(
										mat=self.X_norm, 
										q_anchor=0.5, 
										t_std=1.0, 
										brnl_prob=0.95,
										min_pres=1,
										rand_state=18,
		)

		n_nans_train1 = np.count_nonzero(np.isnan(train1))
		n_nans_train2 = np.count_nonzero(np.isnan(train2))
		n_nans_train3 = np.count_nonzero(np.isnan(train3))

		n_nans_val1 = np.count_nonzero(np.isnan(val1))
		n_nans_val2 = np.count_nonzero(np.isnan(val2))
		n_nans_val3 = np.count_nonzero(np.isnan(val3))

		assert n_nans_train3 > n_nans_train2 > n_nans_train1
		assert n_nans_val1 > n_nans_val2 > n_nans_val3

	def test_mse_func(self):
		"""
		Confirm that the MSE function w/in the util module is 
		working the way that it should be. 
		"""
		# the answer here should be 16
		mat1 = np.array([1,2,3,4]).reshape(2,2)
		mat2 = np.array([5,6,7,8]).reshape(2,2)
		ans1 = mse_func(mat1, mat2)
		assert ans1 == 16

		# the answer here should still be 16
		mat1 = np.float32(mat1)
		mat2 = np.float32(mat2)

		mat1[0,0] = np.nan
		mat2[0,1] = np.nan
		ans2 = mse_func(mat1, mat2)
		assert ans2 == 16

		# a third test. Using PyTorch's MSELoss function to verify
		rng = np.random.default_rng(seed=42)

		mat1 = rng.normal(loc=10, scale=4, size=(5,4))
		mat2 = rng.normal(loc=25, scale=5, size=(5,4))

		mat1_t = torch.tensor(mat1)
		mat2_t = torch.tensor(mat2)

		ans1 = mse_func(mat1, mat2)

		pytorch_mse = torch.nn.MSELoss(reduction="mean")
		ans2 = pytorch_mse(mat1_t, mat2_t)

		assert np.isclose(ans1, ans2, atol=0.01)
