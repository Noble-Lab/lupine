"""
TEST_DATALOADER

Unit tests for the dataset loader class: FactorizationDataset.
Here we're testing on both a small simulated matrix and a small, 
downsampled peptide quants matrix. 
"""
import sys
import unittest
import pytest
import torch
import math
from sklearn.utils import shuffle
import numpy as np
import pandas as pd

sys.path.append("../bin")

from data_loaders import FactorizationDataset
from utils import *

class StandardLoaderTester(unittest.TestCase):
	"""
	Some basic tests for the `FactorizationDataset` class;
	specifically for the non-biased batch selection loader. We'll
	generate a small matrix and use that for testing. 
	"""
	@classmethod
	def setUpClass(self):
		# dataset params
		n_elms = 128
		mv_elms = 40 # how many mvs in the initial matrix?
		n_holdout = 20 # how many elms to hold out for validation?
		n_rows = 16
		n_cols = 8
		rand_seed = 18

		# data loader params
		self.batch_size = 64

		# init numpy's random generator
		rng = np.random.default_rng(seed=rand_seed)

		# the initial random array
		arr = rng.random(n_elms)

		# introduce initial missingness
		arr[0:mv_elms] = np.nan

		# create separate train and test sets
		train = arr.copy()
		val = arr.copy()

		# hold out addn'l entries from test set
		train[mv_elms: mv_elms + n_holdout] = np.nan
		val[mv_elms + n_holdout:] = np.nan

		# shuffle, in the same manner
		train = shuffle(train, random_state=rand_seed)
		val = shuffle(val, random_state=rand_seed)

		# reshape
		self.train = train.reshape(n_rows, n_cols)
		self.val = val.reshape(n_rows, n_cols)

		# convert to tensors
		self.train_t = torch.tensor(self.train)
		self.val_t = torch.tensor(self.val)

		self.train_loader = FactorizationDataset(
								self.train, 
								self.val,
								partition="Train",
								batch_size=self.batch_size,
								biased=False,
		)
		self.val_loader = FactorizationDataset(
								self.train,
								self.val, 
								partition="Valid",
								batch_size=self.batch_size,
								biased=False,
		)
		self.train_loader.get_standard_loader()
		self.val_loader.get_standard_loader()

	def test_batch_size(self):
		"""
		Make sure the data loader `batch_size` and `n_batches` 
		params are working properly. Is the training/validation set
		in fact being split into mini-batches? 
		"""
		# for the setUpClass loader. Standard batch size of 128
		assert self.train_loader.batch_size == self.batch_size
		assert self.train_loader.n_batches == 1
		assert self.val_loader.batch_size == self.batch_size
		assert self.val_loader.n_batches == 1

		# a smaller batch size
		self.train_loader1 = FactorizationDataset(
								self.train, 
								self.val,
								partition="Train",
								batch_size=32,
								biased=False,
		)
		self.val_loader1 = FactorizationDataset(
								self.train,
								self.val,
								partition="Valid",
								batch_size=32,
								biased=False,
		)
		self.train_loader1.get_standard_loader()
		self.val_loader1.get_standard_loader()
		
		assert self.train_loader1.batch_size == 32
		assert self.train_loader1.n_batches == 2
		assert self.val_loader1.batch_size == 32
		assert self.val_loader1.n_batches == 1

		# very large batch size
		self.train_loader2 = FactorizationDataset(
								self.train, 
								self.val, 
								partition="Train",
								batch_size=1000,
								biased=False,
		)
		self.val_loader2 = FactorizationDataset(
								self.train, 
								self.val, 
								partition="Valid",
								batch_size=1000,
								biased=False,
		)
		self.train_loader2.get_standard_loader()
		self.val_loader2.get_standard_loader()

		assert self.train_loader2.batch_size == 128
		assert self.train_loader2.n_batches == 1
		assert self.val_loader2.batch_size == 128
		assert self.val_loader2.n_batches == 1

		# very small batch size, prime number
		self.train_loader3 = FactorizationDataset(
								self.train, 
								self.val, 
								partition="Train",
								batch_size=5,
								biased=False,
		)
		self.val_loader3 = FactorizationDataset(
								self.train, 
								self.val, 
								partition="Valid",
								batch_size=5,
								biased=False,
		)
		self.train_loader3.get_standard_loader()
		self.val_loader3.get_standard_loader()

		assert self.train_loader3.batch_size == 5
		assert self.train_loader3.n_batches == 13
		assert self.val_loader3.batch_size == 5
		assert self.val_loader3.n_batches == 4

	def test_loader_contents(self):
		"""
		Make sure only the non-missing elements have been 
		incorporated into the train and validation loaders. 
		This also implicitly tests that the matrix indices in 
		locs are in fact indices of `train` & `val`. And of 
		course we're testing the loader __iter__ method as well.
		"""
		all_locs_train = torch.cat(self.train_loader.locs, axis=1)
		all_locs_val = torch.cat(self.val_loader.locs, axis=1)

		# assert that all of the items in the train loader are
			# non missing
		for tloc in tuple(all_locs_train.T):
			idx = np.int32(tloc[0]), np.int32(tloc[1])
			assert not torch.isnan(self.train_t[idx])

		# same thing, for the validation loader
		for tloc in tuple(all_locs_val.T):
			idx = np.int32(tloc[0]), np.int32(tloc[1])
			assert not torch.isnan(self.val_t[idx])

		# are there an equal number of matrix indices as targets?
		for locs, target in self.train_loader:
			assert len(locs) == len(target)

		# does the train loader's `mat` match up with `train`?
		loader_mat_t = pd.DataFrame(
							np.float32(self.train_loader.mat))
		train_mat = pd.DataFrame(
							np.float32(self.train_t))
		assert loader_mat_t.equals(train_mat)

		# does the validation loader's `mat` match up with `train`?
		loader_mat_v = pd.DataFrame(
							np.float32(self.val_loader.mat))
		val_mat = pd.DataFrame(
							np.float32(self.val_t))
		assert loader_mat_v.equals(val_mat)

	def test_nonzero_elements(self):
		"""
		Make sure the loaders have the expected number of non-missing
		elements. 
		"""
		for locs, target in self.train_loader:
			pass
		n_xij_train = np.count_nonzero(~np.isnan(target))

		for locs, target in self.val_loader:
			pass
		n_xij_val = np.count_nonzero(~np.isnan(target))

		gt_xij_train = np.count_nonzero(~np.isnan(self.train))
		gt_xij_val = np.count_nonzero(~np.isnan(self.val))

		assert n_xij_train == gt_xij_train
		assert n_xij_val == gt_xij_val

class BiasedDataLoaderTester(unittest.TestCase):
	"""
	Testing out the biased and non-biased data loader functions
	as part of the `FactorizationDataset` class. Here were using
	a simulated matrix, but with much higher mean and std. 
	"""
	@classmethod
	def setUpClass(self):
		"""
		Set up a small simulated matrix with mean and standard
		deviation 1000. Partition using MNAR and MCAR procedures
		"""
		# data simulation params
		mean=1000
		std=1000
		n_rows=32
		n_cols=16
		seed=18

		# partitioning params
		quantile_anchor=0.35
		part_std=1.4
		b_prob=0.9
		min_present=1
		val_frac=0.1
		test_frac=0.1

		# init the pseudorandom number generator
		rng = np.random.default_rng(seed=seed)

		# simulate a normal distribution
		mat = np.matrix(
				np.abs(
					rng.normal(
						loc=mean, 
						scale=std, 
						size=(n_rows, n_cols),
					)
				)
		)
		mat = np.array(mat)

		# MNAR partition
		self.train_mnar, self.val_mnar, test_mnar = \
			mnar_partition_thresholds_matrix(
				mat,
				q_anchor=quantile_anchor,
				t_std=part_std, 
				brnl_prob=b_prob,
				min_pres=min_present, 
				rand_state=seed,
		)
		# MCAR partition 
		self.train_mcar, self.val_mcar, test_mcar = \
			mcar_partition(
				mat, 
				val_frac=val_frac, 
				test_frac=test_frac, 
				min_present=min_present, 
				random_state=seed,
		)

	def test_non_biased_loaders(self):
		"""
		Generate `FactorizationDataset` loaders in a non-biased
		manner, from the MCAR partitioned quants. Make sure that
		the means of the train and valid loaders are not far
		off. Setting `testing` to True here will freeze PyTorch's 
		random seed in the `FactorizationDataset` class. 
		"""
		train_loader = FactorizationDataset(
								X=self.train_mcar, 
								X_val=self.val_mcar, 
								partition="Train",
								biased=False,
								rand_seed=42,
		)
		val_loader = FactorizationDataset(
								X=self.train_mcar, 
								X_val=self.val_mcar, 
								partition="Valid",
								biased=False,
								rand_seed=42,
		)
		train_loader.get_standard_loader()
		val_loader.get_standard_loader()

		with pytest.raises(AssertionError):
			train_loader.get_biased_loader()
			val_loader.get_biased_loader()

		train_target_size_exp = np.ceil((32*16) * 0.8)
		val_target_size_exp = np.floor((32*16) * 0.1)

		# get the train set targets
		for locs, train_targets in train_loader:
			pass
		# get the valid set targets
		for locs, val_targets in val_loader:
			pass

		assert len(train_targets) == train_target_size_exp
		assert len(val_targets) == val_target_size_exp

		train_loader_mean = torch.mean(train_targets)
		val_loader_mean = torch.mean(val_targets)
		assert math.isclose(
					train_loader_mean, 
					val_loader_mean, 
					abs_tol=50
		)

	def test_biased_loaders_MCAR(self):
		"""
		Used the biased mini-batch selection procedure to 
		generate mini-batches. The means of the valid set
		should be lower than the train set. Here we're doing this
		with the MCAR partition procedure. So the mini-batch
		selection procedure is the only thing driving the mean
		of the validation set values down. 
		"""
		train_loader = FactorizationDataset(
								X=self.train_mcar, 
								X_val=self.val_mcar, 
								partition="Train",
								biased=True,
								testing=True,
								rand_seed=18,
		)
		val_loader = FactorizationDataset(
								X=self.train_mcar, 
								X_val=self.val_mcar, 
								partition="Valid",
								biased=True,
								testing=True,
								rand_seed=36,
		)
		with pytest.raises(AssertionError):
			train_loader.get_standard_loader()
			val_loader.get_standard_loader()

		train_loader.get_biased_loader()
		val_loader.get_biased_loader()

		# get the train set targets
		for locs, train_targets in train_loader:
			pass
		# get the valid set targets
		for locs, val_targets in val_loader:
			pass

		train_loader_mean = torch.mean(train_targets)
		val_loader_mean = torch.mean(val_targets)
		assert train_loader_mean > val_loader_mean
		assert not math.isclose(
							train_loader_mean, 
							val_loader_mean, 
							abs_tol=100,
		)
		# the biased batch selection procedure should be pulling
		# 	the train loader mean down relative to the general
		#	training set mean. 
		train_mean = np.nanmean(self.train_mcar)
		assert train_loader_mean < train_mean

	def test_biased_loaders_MNAR(self):
		"""
		Used the biased mini-batch selection procedure to 
		generate mini-batches, for the MNAR partitioned quants. 
		Here the difference between train, train mini batch and
		valid mini batch means should be more extreme than the
		MCAR test. 
		"""
		train_loader = FactorizationDataset(
								X=self.train_mnar, 
								X_val=self.val_mnar, 
								partition="Train",
								biased=True,
								rand_seed=42,
		)
		val_loader = FactorizationDataset(
								X=self.train_mnar, 
								X_val=self.val_mnar, 
								partition="Valid",
								biased=True,
								rand_seed=42,
		)
		with pytest.raises(AssertionError):
			train_loader.get_standard_loader()
			val_loader.get_standard_loader()

		train_loader.get_biased_loader()
		val_loader.get_biased_loader()

		# get the train set targets
		for locs, train_targets in train_loader:
			pass
		# get the valid set targets
		for locs, val_targets in val_loader:
			pass

		train_loader_mean = torch.mean(train_targets)
		val_loader_mean = torch.mean(val_targets)
		assert train_loader_mean > val_loader_mean
		assert not math.isclose(
							train_loader_mean, 
							val_loader_mean, 
							abs_tol=300
		)
		# the biased batch selection procedure should be pulling
		# 	the train loader mean down relative to the general
		#	training set mean. 
		train_mean = np.nanmean(self.train_mnar)
		assert train_loader_mean < train_mean

	def test_return_missing(self):
		"""
		Testing the case where we ask the data loader to return
		the missing values, as we do in `base::transform`. 
		"""
		eval_loader = FactorizationDataset(
									X=self.train_mnar, 
									X_val=self.val_mnar, 
									partition="Train",
									biased=False,
									missing=True,
		)
		eval_loader.get_standard_loader()

		for locs, targets in eval_loader:
			pass 
		assert np.count_nonzero(np.isnan(targets)) == len(targets)

		targets = []
		for loc in locs: 
			target = self.train_mnar[tuple(loc)]
			targets.append(target)
		assert np.count_nonzero(np.isnan(targets)) == len(targets)

	def test_minibatch_stochasticity_nonbiased(self):
		"""
		Confirm that the mini-batch selection process is indeed
		stochastic, that is, two different data loaders should 
		contain different elements. The non-biased version here.
		We expect to see a lot more overlap between independent 
		data loaders when we're working with the non-biased
		mini-batch selection procedure
		"""
		loader1 = FactorizationDataset(
							X=self.train_mnar, 
							X_val=self.val_mnar, 
							partition="Train",
							biased=False,
		)
		loader1.get_standard_loader()

		loader2 = FactorizationDataset(
							X=self.train_mnar, 
							X_val=self.val_mnar, 
							partition="Train",
							biased=False,
		)
		loader2.get_standard_loader()

		for locs1, targets1 in loader1:
			pass

		for locs2, targets2 in loader2:
			pass

		targets1 = np.array(targets1)
		targets2 = np.array(targets2)

		assert not np.array_equal(targets1, targets2)

		# because this is such a small test set, all of the 
		# 	present values will end up in both data loaders
		inter = np.intersect1d(targets1, targets2)
		assert len(inter) == len(targets1)

	def test_minibatch_stochasticity_biased(self):
		"""
		Confirm that the mini-batch selection process is indeed
		stochastic, that is, two different data loaders should 
		contain different elements. The biased version -- should
		be way less overlap between independent data loaders. 
		"""
		loader1 = FactorizationDataset(
							X=self.train_mnar, 
							X_val=self.val_mnar, 
							partition="Train",
							biased=True, 
		)
		loader1.get_biased_loader()

		loader2 = FactorizationDataset(
							X=self.train_mnar, 
							X_val=self.val_mnar, 
							partition="Train",
							biased=True, 
		)
		loader2.get_biased_loader()

		for locs1, targets1 in loader1:
			pass

		for locs2, targets2 in loader2:
			pass

		targets1 = np.array(targets1)
		targets2 = np.array(targets2)

		assert not np.array_equal(targets1, targets2)

		inter = np.intersect1d(targets1, targets2)
		assert len(inter) < len(targets1)
		assert len(inter) < len(targets2)

	def test_biased_batch_sizes(self):
		"""
		For the biased batch selection procedure, make sure that the 
		data loader is generating the expected number of mini-batches 
		and that each mini-batch contains the expected number of non-
		missing elements. 
		"""
		train_loader = FactorizationDataset(
							X=self.train_mnar, 
							X_val=self.val_mnar, 
							partition="Train",
							batch_size=128,
							biased=True, 
							rand_seed=36,
		)
		train_loader.get_biased_loader()

		val_loader = FactorizationDataset(
							X=self.train_mnar, 
							X_val=self.val_mnar, 
							partition="Valid",
							batch_size=128,
							biased=False, 
							rand_seed=36,
		)
		val_loader.get_standard_loader()

		assert train_loader.n_batches == 3
		assert val_loader.n_batches == 1

		train_batch_sizes = []
		for locs, target in train_loader:
			batch_size = len(target)
			train_batch_sizes.append(batch_size)

		for locs, target in val_loader:
			batch_size_val = len(target)

		assert train_batch_sizes == [123, 123, 123]
		assert batch_size_val == 66

# class DataLoaderTesterQuants(unittest.TestCase):
# 	"""
# 	Here we're expanding out `FactorizationDataset` class test
# 	suite to a small peptide quants matrix. 
# 	"""
# 	@classmethod
# 	def setUpClass(self):
# 		# read in the peptide quants test file
# 		TESTFILE = "peptide_quants_tester.csv"
# 		test_file = pd.read_csv(TESTFILE)
# 		quants = np.array(test_file)
