"""
TEST_SCALERS

Unit tests for the scaler classes: STDScaler and MinMaxScaler. Here
we're combining tests on a simulated matrix with tests from a small
peptide quants matrix. The reason for this is that I've realized the
distribution of real peptide quants matrices is super different 
than our simulated matrices, in terms of mean and std. So I think 
including some tests on a small peptide quants matrix is really
the way to go. 
"""
import unittest
import pytest
import torch
import numpy as np
import pandas as pd
import warnings
from sys import path

path.append("../bin")
from utils import *
from scalers import STDScaler, MinMaxScaler

class STDScalerTester(unittest.TestCase):
	"""
	Test the `STDScaler` class on a small simulated matrix of known
	dimensionality. 
	"""
	@classmethod
	def setUpClass(self):
		"""
		A class object to test the STDScaler class. Generate a small
		test matrix. 
		"""
		W = np.matrix([1.0, 3.0, 5.0])
		H = np.matrix([7.0, 9.0])

		X = np.matmul(W.T, H)

		X_mv = X.copy()
		X_mv[2,1] = np.nan

		self.X = torch.tensor(X)
		self.X_mv = torch.tensor(X_mv)

		self.scaler = STDScaler(testing=True, log_transform=False)

		# init numpy's random number generator
		rand_seed = 18
		self.rng = np.random.default_rng(seed=rand_seed)
	
	def test_STDScaler(self):
		"""
		Make sure the STDScaler scaled values match the expectation.
		Note that for this example, you actually get different 
		results when computing the standard deviation with Numpy
		vs PyTorch.
		"""
		X_mean = torch.mean(self.X)
		X_std = torch.std(self.X)

		X_scaled = self.X / X_std
		X_STDScaled = self.scaler.fit_transform(
										X=self.X, 
										X_val=self.X_mv, 
										partition="Train",
		)
		X_scaled_pd = pd.DataFrame(np.array(X_scaled))
		X_STDScaled_pd = pd.DataFrame(np.array(X_STDScaled))
		assert X_scaled_pd.equals(X_STDScaled_pd)

	def test_STDScaler_mv(self):
		"""
		Make sure the STDScaler scaled values match the expectation
		in the case where X has missing values. 
		"""
		missing = torch.isnan(self.X_mv)

		X_mv_mean = torch.mean(self.X_mv[~missing])
		X_mv_std = torch.std(self.X_mv[~missing])
		
		X_mv_scaled = self.X_mv / X_mv_std
		X_mv_std_scaled = self.scaler.fit_transform(
										X=self.X_mv,
										X_val=None,
										partition="Train",
		)
		X_mv_scaled_pd = pd.DataFrame(np.array(X_mv_scaled))
		X_mv_std_scaled_pd = pd.DataFrame(np.array(X_mv_std_scaled))

		assert X_mv_scaled_pd.equals(X_mv_std_scaled_pd)

	def test_complete_mat_vs_noncomplete(self):
		"""
		The STDScaler should return different scaled matrices for 
		`X` and `X_mv`. 
		"""
		X_scaled = self.scaler.fit_transform(
										X=self.X, 
										X_val=None, 
										partition="Train",
		)
		X_mv_scaled = self.scaler.fit_transform(
										X=self.X_mv, 
										X_val=None, 
										partition="Train",
		)
		X_scaled[2,1] = np.nan

		X_scaled_pd = pd.DataFrame(np.array(X_scaled))
		X_mv_scaled_pd = pd.DataFrame(np.array(X_mv_scaled))

		assert X_scaled_pd.equals(X_mv_scaled_pd) == False

	def test_inverse_transform(self):
		"""
		Make sure the STDScaler class's inverse scaling method
		is working
		"""
		# for the complete matrix
		X_scaled = self.scaler.fit_transform(
									X=self.X,
									X_val=None,
									partition="Train",
		)
		X_inv_scaled = self.scaler.inverse_transform(
									X=X_scaled,
									X_val=None, 
									partition="Eval",
		)
		X_pd = pd.DataFrame(np.array(self.X))
		X_inv_scaled_pd = pd.DataFrame(np.array(X_inv_scaled))

		assert X_pd.equals(X_inv_scaled_pd)

		# for the missing value matrix
		X_mv_scaled = self.scaler.fit_transform(
									X=self.X_mv,
									X_val=None,
									partition="Train",
		)
		X_mv_inv_scaled = self.scaler.inverse_transform(
									X=X_mv_scaled,
									X_val=None,
									partition="Train",
		)
		X_mv_pd = pd.DataFrame(np.float32(self.X_mv))
		X_mv_inv_scaled_pd = \
				pd.DataFrame(np.float32(X_mv_inv_scaled))

		assert X_mv_pd.equals(X_mv_inv_scaled_pd)

	def test_all_mvs(self):
		"""
		Make sure the STDScaler class gracefully handles the case 
		where the input matrix is all NaNs or all zeros
		"""
		arr = np.zeros((6,5))
		nan_arr = arr.copy()

		nan_arr[nan_arr == 0] = np.nan

		Z_nan = self.scaler.fit_transform(X=nan_arr, X_val=None)
		Z_zero = self.scaler.fit_transform(X=arr, X_val=None)

		assert torch.isnan(Z_nan).all()
		assert torch.isnan(Z_zero).all()

	def test_non_tensor(self):
		"""
		Make sure the STDScaler class can handle non-torch.tensor
		input matrices
		"""
		X = self.rng.random((6,5))
		Y = self.scaler.fit_transform(X=np.array(X), X_val=None)

		assert torch.is_tensor(Y)
		assert Y.shape == (6,5)

class MinMaxScalerTester(unittest.TestCase):
	"""
	Test the `MinMaxScaler` on a small simulated matrix of known
	latent dimensionality. 
	"""
	@classmethod
	def setUpClass(self):
		"""
		A class object to test the MinMaxScaler class. Generate a 
		small test matrix. 
		"""
		W = np.matrix([1.0, 3.0, 5.0])
		H = np.matrix([7.0, 9.0])

		self.X = np.matmul(W.T, H)

		self.X_mv = self.X.copy()
		self.X_mv[2,1] = np.nan

		self.X_t = torch.tensor(self.X)
		self.X_t_mv = torch.tensor(self.X_mv)

		self.scaler = MinMaxScaler()
	
	def test_MinMaxRecon(self):
		"""
		Make sure the Min/Max scaled values match the expectation
		"""
		# scale by hand
		X_min = np.nanmin(self.X_t)
		X_max = np.nanmax(self.X_t)
		Y = (self.X_t - X_min) / (X_max - X_min)

		# scale with the MinMaxScaler
		Z = self.scaler.fit_transform(self.X_t)

		Y_pd = pd.DataFrame(np.float32(Y))
		Z_pd = pd.DataFrame(np.float32(Z))

		assert Y_pd.equals(Z_pd)

		# by definition, min max scaling puts everything on a scale
			# between 0 and 1
		assert np.nanmax(Y) == 1
		assert np.nanmax(Z) == 1
		assert np.nanmin(Y) == 0
		assert np.nanmin(Z) == 0

	def test_input_format(self):
		"""
		Assert that the Min Max scaling method works even when
		I give it a non-tensor as input, in this case a numpy
		array. 
		"""
		Y = self.scaler.fit_transform(np.array(self.X))
		assert torch.is_tensor(Y)

		# scale by hand
		X_min = np.nanmin(self.X)
		X_max = np.nanmax(self.X)
		Z = (self.X - X_min) / (X_max - X_min)

		# make sure the MinMaxScaler scaled matrix looks the way
			# it's supposed to
		Y_pd = pd.DataFrame(np.float32(Y))
		Z_pd = pd.DataFrame(np.float32(Z))

		assert Y_pd.equals(Z_pd)

	def test_mv(self):
		"""
		Make sure the MinMaxScaler properly scales input matrices
		that contain MVs
		"""
		# scale by hand
		X_min = np.nanmin(self.X_t_mv)
		X_max = np.nanmax(self.X_t_mv)
		Y = (self.X_t_mv - X_min) / (X_max - X_min)

		# with the MinMaxScaler
		Z = self.scaler.fit_transform(self.X_t_mv)

		Y_pd = pd.DataFrame(np.float32(Y))
		Z_pd = pd.DataFrame(np.float32(Z))

		assert Y_pd.equals(Z_pd)
		assert np.isnan(Y[2,1])
		assert np.isnan(Z[2,1]) 

		assert np.nanmax(Y) == 1
		assert np.nanmax(Z) == 1
		assert np.nanmin(Y) == 0
		assert np.nanmin(Z) == 0

	def test_all_mvs(self):
		"""
		Make sure the MinMaxScaler method gracefully handles the 
		case where the input matrix is all NaNs, or all zeros. 
		"""
		arr = np.zeros((6,5))
		nan_arr = arr.copy()

		nan_arr[nan_arr == 0] = np.nan

		with self.assertWarns(RuntimeWarning):
			Z = self.scaler.fit_transform(nan_arr)

		Z = self.scaler.fit_transform(arr)
		assert torch.isnan(Z).all()

	def test_inverse_transform(self):
		"""
		Make sure that the MinMaxScaler method can properly 
		reconstruct the input matrix after scaling
		"""
		# the fully present matrix case
		X_scaled = self.scaler.fit_transform(self.X_t)
		X_inv_scaled = self.scaler.inverse_transform(X_scaled)

		X_inv_scaled_pd = pd.DataFrame(np.float32(X_inv_scaled))
		X_pd = pd.DataFrame(np.float32(self.X_t))

		assert X_pd.equals(X_inv_scaled_pd)

		# the matrix with missing value case
		Xmv_scaled = self.scaler.fit_transform(self.X_t_mv)
		Xmv_inv_scaled = self.scaler.inverse_transform(Xmv_scaled)

		Xmv_inv_scaled_pd = pd.DataFrame(np.float32(Xmv_inv_scaled))
		Xmv_pd = pd.DataFrame(np.float32(self.X_t_mv))

		assert Xmv_pd.equals(Xmv_inv_scaled_pd)

class STDTesterQuantsData(unittest.TestCase):
	"""
	Actually testing this bad boy with real peptide quants data.
	The reason we're doing this is because the distribution of the
	quants matrices is kinda wacky and doesn't totally resemble
	our simulated matrices. For example, in the peptide quant mat, 
	std > mean. 
	"""
	@classmethod
	def setUpClass(self):
		# define some params
		quantile_anchor=0.3
		part_std=0.6
		b_prob=0.65
		min_present=4
		valid_fraction=0.1
		test_fraction=0.1
		seed=18

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
		# init the pseudorandom number generator
		self.rng = np.random.default_rng(seed=seed)

	def test_train_scaling(self):
		"""
		Make sure the `STDScaler` scaling procedure recapitulates the
		'by hand' scaling procedure, for the training set. 
		"""
		# SCALE WITH THE `STDSCALER`
		scaler = STDScaler(testing=False, log_transform=True)
		scaler_trans_train = scaler.fit_transform(
										X=self.train_mnar, 
										X_val=self.val_mnar, 
										partition="Train",
		)
		# SCALE BY HAND
			# exclude background noise values
		train = self.train_mnar.copy()
		train[train <= 1.0] = np.nan

		# log transform
		train_log = torch.log(torch.tensor(train))

		# quantile based outlier exclusion
		qt01 = torch.nanquantile(train_log, q=0.001)
		train_log[train_log <= qt01] = np.nan
		    
		# get the std of the non-missing tensor elements    
		nonmissing = train_log[~torch.isnan(train_log)]
		trans_train = train_log / torch.std(nonmissing)

		trans_train_pd = pd.DataFrame(np.float32(trans_train))
		scaler_trans_pd = pd.DataFrame(np.float32(scaler_trans_train))
		assert trans_train_pd.equals(scaler_trans_pd)

	def test_valid_scaling(self):
		"""
		Make sure the `STDScaler` scaling procedure recapitualtes the
		'by hand' scaling procedure, this time for the validation set. 
		This is important because we should always be calculating the
		scaling factor on the training set, and not the validation
		set. 
		"""
		# USE THE `STDSCALER` TO SCALE
		scaler = STDScaler(testing=False, log_transform=True)
		scaler_trans_val = scaler.fit_transform(
											X=self.train_mnar, 
											X_val=self.val_mnar, 
											partition="Valid",
		)
		# SCALE BY HAND
		# exclude background noise values
		train = self.train_mnar.copy()
		val = self.val_mnar.copy()
		train[train <= 1.0] = np.nan
		val[val <= 1.0] = np.nan

		# log transform
		train_log = torch.log(torch.tensor(train))
		val_log = torch.log(torch.tensor(val))

		# quantile based outlier exclusion 
		qt01 = torch.nanquantile(train_log, q=0.001)
		train_log[train_log <= qt01] = np.nan
		val_log[val_log <= qt01] = np.nan

		# get the scaling factor
		nonmissing = train_log[~torch.isnan(train_log)]
		train_std = torch.std(nonmissing)

		val_scaled = val_log / train_std

		# test equality
		scaler_trans_pd = pd.DataFrame(np.float32(scaler_trans_val))
		trans_val_pd = pd.DataFrame(np.float32(val_scaled))
		assert scaler_trans_pd.equals(trans_val_pd)

	def test_non_log_scaling(self):
		"""
		Make sure the `STDScaler` scaling procedure recapitualtes 
		the 'by hand' scaling procedure, for the validation set, 
		when we set `log_transform` to False. 
		"""
		# USE THE `STDSCALER` TO SCALE
		scaler = STDScaler(testing=False, log_transform=False)
		scaler_trans_val = scaler.fit_transform(
											X=self.train_mnar, 
											X_val=self.val_mnar, 
											partition="Valid",
		)
		# SCALE BY HAND
		# exclude background noise values
		train = self.train_mnar.copy()
		val = self.val_mnar.copy()
		train[train <= 1.0] = np.nan
		val[val <= 1.0] = np.nan

		train = torch.tensor(train)
		val = torch.tensor(val)

		# quantile based outlier exclusion 
		qt01 = torch.nanquantile(train, q=0.001)
		train[train <= qt01] = np.nan
		val[val <= qt01] = np.nan

		# get the scaling factor
		nonmissing = train[~torch.isnan(train)]
		train_std = torch.std(nonmissing)

		val_scaled = val / train_std

		# test equality
		scaler_trans_pd = pd.DataFrame(np.float32(scaler_trans_val))
		trans_val_pd = pd.DataFrame(np.float32(val_scaled))
		assert scaler_trans_pd.equals(trans_val_pd)

	def test_minimal_noise_filter(self):
		"""
		Make sure the whole filter out matrix entries less than or 
		equal to one thing is working, and that this param can be
		toggled on or off. Assert that for a matrix in which every
		X_ij < 1.0, it returns a complete NaN matrix. 
		"""
		arr = np.ones((12,6))
		scaler = STDScaler(testing=False, log_transform=False)
		scaled = scaler.fit_transform(
							X=arr, 
							X_val=None, 
							partition="Train",
		)
		arr[:] = np.nan

		# check equality
		arr_pd = pd.DataFrame(arr)
		scaled_pd = pd.DataFrame(np.array(scaled))
		assert arr_pd.equals(scaled_pd)

		# simulate a matrix. All of these values should 
		#    be between 0 and 1
		sim = np.array(self.rng.random(120))
		sim = sim.reshape((12,10))
		# scale, where `testing` is True
		scaler1 = STDScaler(testing=True, log_transform=False)
		scaled1 = scaler1.fit_transform(
								X=sim, 
								X_val=None, 
								partition="Train",
		)
		# scale, where `testing` is False
		scaler2 = STDScaler(testing=False, log_transform=False)
		scaled2 = scaler2.fit_transform(
								X=sim, 
								X_val=None, 
								partition="Train",
		)

		scaled1_nans = np.nonzero(np.isnan(scaled1))
		assert len(scaled1_nans) == 0 

		scaled2_nans = np.nonzero(np.isnan(scaled2))
		assert len(scaled2_nans) == 120

	def test_mcar_scaling(self):
		"""
		Making sure that nothing funky happens when I scale a 
		matrix that has been partitioned with MCAR. 
		"""
		scaler1 = STDScaler(testing=False, log_transform=True)
		scaler2 = STDScaler(testing=False, log_transform=True)

		train_scaled = scaler1.fit_transform(
									X=self.train_mcar, 
									X_val=self.val_mcar, 
									partition="Train",
		)
		val_scaled = scaler2.fit_transform(
									X=self.train_mcar, 
									X_val=self.val_mcar, 
									partition="Valid",
		)
		assert scaler1.std == scaler2.std
		assert scaler1.mean == scaler2.mean

		assert train_scaled.shape == (886, 32)
		assert val_scaled.shape == (886, 32)

		train_scaled_pd = pd.DataFrame(np.array(train_scaled))
		val_scaled_pd = pd.DataFrame(np.array(val_scaled))
		assert not train_scaled_pd.equals(val_scaled_pd)
