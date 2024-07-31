"""
DATA-LOADERS

This module contains the FactorizationDataset class, which is used
to set up mini-batches for the train and validation sets. This 
is a custom data loader specifically designed for matrix 
factorization problems, and specifically tailored for MNAR settings.
Moving this to a separate file because `base` was getting unwieldy, 
and so I can easily add methods and configs to this class. 
"""
import torch 
import numpy as np
import pandas as pd

class FactorizationDataset():
	"""
	A custom PyTorch Dataset class designed to set up input 
	matrices for factorization with our PyTorch imputation models. 

	Parameters
	----------
	X : np.ndarray, 
		The matrix to factorize. Should be the training set
	X_val : np.ndarray,
		The validation set. We won't necessarily use this set.
		Can be None.
	partition : str, 
		The disjoint set. {"Train", "Valid", "Test"}. This tells
		us which set to make the data loader from
	batch_size :  int, optional, 
		The batch size. `None` uses the full dataset.
	biased : bool, optional, 
		Use the MNAR biased mini-batch selection method?
	shuffle : bool, optional, 
		Shuffle the order of the returned examples
	missing : bool, optional, 
		If `True` the missing elements are returned
	testing : bool, 
		Is the model being run in testing mode? If yes, random seeds
		will be set manually
	q_anchor : float, optional, for MNAR biased selection
		What quantile of X to center the thresholds distribution
	t_std : float, optional, for MNAR biased selection
		The standard deviation of the thresholds distribution
	brnl_prob : float, optional, for MNAR biased selection
		The success probability for the Bernoulli trial 
	n_iters : int, optional, for MNAR biased selection
		The number of iterations to perform the biased selection 
		loop for
	seed : int, optional, for MNAR biased selection
		The random seed. The default is to not assign a random seed, 
		which tells numpy to select on on its own. This ensures non-
		deterministic mini-batch selection. Note that setting this 
		seed doesn't necessarily ensure the same locs end up in the
		mini-batches. 
	device : str, optional,
		The device to store pytorch tensors on. {"cpu", "cuda"}
	"""
	def __init__(
		self, 
		X, 
		X_val,
		partition,
		batch_size=None, 
		biased=True,
		shuffle=True, 
		missing=False,
		testing=False,
		anchor=0.2,
		std=0.7,
		b_prob=0.8,
		iters=3,
		rand_seed=None, 
		device="cpu",
	):
		X = _check_tensor(X)
		if X_val is not None: 
			X_val = _check_tensor(X_val)

		self.testing = testing
		self.part = partition
		self.biased = biased 
		self.shuffle = shuffle
		self.missing = missing 
		self.batch_size = batch_size
		self.n_batches = 0
		self.locs = None
		self.device = device 	

		if self.part == "Train":
			self.mat = X
		elif self.part == "Valid":
			self.mat = X_val

		if self.biased:
			self.anchor = anchor
			self.std = std 
			self.b_prob = b_prob 
			self.iters = iters 
			self.rand_seed = rand_seed 
			# get the quantiles and the std of the training set
			self.q_thresh = np.nanquantile(X, self.anchor)
			self.quants_std = np.nanstd(X)
		else:
			self.anchor = None
			self.std = None
			self.b_prob = None
			self.iters = None 
			self.rand_seed = None

		# this will ensure stochastic data loaders. good for testing,
		#   but should be turned off for everything else 
		if rand_seed:
			torch.manual_seed(rand_seed)
			self.rand_seed = rand_seed

	def get_standard_loader(self):
		"""
		Get the mini-batch data loader, in the `standard` manner in 
		which all matrix X_ijs are assigned to mini-batches with 
		equal probability. 	
		"""
		assert not self.biased, "You did an oopsie"

		# set the batch size
		if self.batch_size is not None:
			if self.batch_size > self.mat.numel():
				self.batch_size = self.mat.numel()
		else:
			self.batch_size = self.mat.numel()

		# get a boolean mask of all of the NaN entries
		selected = torch.isnan(self.mat)

		# reverse the mask -- True for present entries
		if not self.missing:
			selected = ~selected

		# get an array of tuples corresponding to present X_ijs
		locs = torch.nonzero(selected)
		# transpose the array of tuples
		locs = locs.T

		# permute the locs
		if self.shuffle:
			locs = \
				locs[:, torch.randperm(locs.shape[1])]

		# get the number of elements per batch
		elms = len(self.mat[selected])

		self.n_batches = np.int32(np.floor(elms / self.batch_size))
		if self.n_batches < 1:
			self.n_batches = 1

		# split into `n_batches` mini-batches. 
		# 	generates a list of mini-batches
		self.locs = \
			torch.tensor_split(locs, self.n_batches, dim=1)

		return

	def get_biased_loader(self):
		"""
		Get the mini-batch data loader, where low abundance quant
		values are preferentially selected for each mini batch. This
		will result in an overrepresentation of certain matrix X_ijs
		"""
		assert self.biased, "You did an oopsie"

		# get a boolean mask of all of the NaN entries
		selected = torch.isnan(self.mat)

		# reverse the mask -- True for present entries
		if not self.missing:
			selected = ~selected
		
		# get an array of tuples corresponding to present X_ijs
		locs = torch.nonzero(selected)
		# transpose the array of tuples
		locs = locs.T

		# permute the locs
		if self.shuffle:
			locs = \
				locs[:, torch.randperm(locs.shape[1])]
		# get the biased mini-batches
		#locs_biased = self._get_biased_mini_batches(locs)
		locs_biased = self._get_biased_mini_batches_vectorized(locs)

		# permute again
		locs_biased = \
			locs_biased[:, torch.randperm(locs_biased.shape[1])]

		# set the batch size
		if self.batch_size is not None:
			if self.batch_size > len(locs_biased[0]):
				self.batch_size = len(locs_biased[0])
		else:
			self.batch_size = len(locs_biased[0])

		# get the number of elements per batch
		elms = len(locs_biased[0])
		self.n_batches = np.int32(np.ceil(elms / self.batch_size))

		# split into `n_batches` mini-batches. Generates a list of 
		#   mini-batches
		self.locs = torch.tensor_split(
						locs_biased, self.n_batches, dim=1)
		return

	def _get_biased_mini_batches_vectorized(self, locs):
		"""
		An alternate procedure for probabilistically selecting matrix
		X_ijs for mini-batches based on their quant value. All of the
		computation here is vectorized, so this routine is extremely
		fast. 

		Parameters
		----------
		locs : torch.tensor, 
			A tensor of tuples indicating the indicies of the present
			values in `mat`

		Returns
		----------
		locs_biased : torch.tensor,
			A tensor array of tuples corresponding to matrix indices
			in X that comprise each mini-batch
		"""
		# init numpy's pseudorandom number generator
		rng = np.random.default_rng(self.rand_seed)

		i_idx_all = np.array([])
		j_idx_all = np.array([])

		for i in range(0, self.iters):
			# set up the thresholds matrix 
			thresh_mat = rng.normal(
								loc=self.q_thresh, 
								scale=(self.quants_std * self.std), 
								size=self.mat.shape,
			)
			# no longer strictly Gaussian
			thresh_mat = np.abs(thresh_mat)

			# Figure out where T_ij > X_ij
			thresh_mask = thresh_mat > self.mat.numpy()
			thresh_mask = np.int32(thresh_mask)

			# Limit to just the X_ijs present in the train set
			thresh_mask = np.float32(thresh_mask)
			thresh_mask[torch.isnan(self.mat)] = np.nan

			# Get Bernoulli trial results for both high and 
			#   low abundance quants
			low_val_binom_res = rng.binomial(
									n=1, 
									p=self.b_prob, 
									size=(thresh_mat.shape),
			)
			high_val_binom_res = rng.binomial(
									n=1, 
									p=1-self.b_prob, 
									size=(thresh_mat.shape),
			)
			# Again, limit to just the X_ijs in the train set
			low_val_binom_res = np.float32(low_val_binom_res)
			high_val_binom_res = np.float32(high_val_binom_res)
			low_val_binom_res[torch.isnan(self.mat)] = np.nan
			high_val_binom_res[torch.isnan(self.mat)] = np.nan

			# Low abundance quant: T_ij > X_ij and Bernoulli success
			low_idx = \
				np.where((thresh_mask==1) & (low_val_binom_res==1))
			# High abundance quant: T_ij < X_ij and Bernoulli success
			high_idx = \
				np.where((thresh_mask==0) & (high_val_binom_res==1))

			i_idx = np.concatenate((low_idx[0], high_idx[0]))
			j_idx = np.concatenate((low_idx[1], high_idx[1]))

			i_idx_all = np.int32(np.concatenate((i_idx_all, i_idx)))
			j_idx_all = np.int32(np.concatenate((j_idx_all, j_idx)))

		locs_biased = torch.tensor(np.vstack((i_idx_all, j_idx_all)))
		locs_biased = locs_biased.long()
		return locs_biased

	def __iter__(self):
		"""
		Return an iterable of both matrix indicies and the matrix 
		entries associated with those indices. 
		"""
		for loc in self.locs:
			yield loc.T, self.mat[tuple(loc)]

	def __len__(self):
		"""
		Return the number of mini-batches
		"""
		return len(self.locs)

# non-class method
def _check_tensor(array):
	"""
	Check that an array is the correct shape and a tensor.
	This function will also coerce `pandas.DataFrame` and
	`numpy.ndarray` objects to `torch.Tensor` objects.

	Parameters
	----------
	array : array-like,
		The input array

	Returns
	-------
	array : torch.Tensor,
		The array as a `torch.Tensor`
	"""
	if not isinstance(array, torch.Tensor):
		array = torch.tensor(array)

	return array
