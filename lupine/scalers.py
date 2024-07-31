"""
SCALERS

This module contains methods to scaler our inputs prior to 
imputation with our models. This module contains three classes:
	1. STDScaler - Scale the input matrix according to 
	the standard deviation of the entire matrix
	2. MinMaxScaler - Min/Max scaling of the matrix
	3. LogFoldScaler - A very simple scalar for the log-fold
	change quantification data for the Imputizer3000 model

The entire workflow seems to do better when I do the scaling outside
of the model code entirely, i.e., before partitioning. 
"""
import torch 
import numpy as np
import pandas as pd

class STDScaler():
	"""
	Scale the matrix according to the standard deviation of the 
	entire matrix. When you call `fit_transform` you have to provide
	both training and validation sets. This is because we want to
	calculate the scaling factors on ONLY the training set, so if
	you want to scale the validation set, you need to have the 
	training set on hand to compute the scaling factor. 

	Parameters
	----------
	testing : bool, optional, 
		Are we in testing mode? If True, we won't perform the
		basic outlier filtering procedure in which values <= 1.0
		are set to NaN. 
	log_transform : bool, optional,
		Log transform the peptide quants prior to scaling? 
	quantile_filter : float, optional, 
		The quantile to filter low values based on. Any value under
		this quantile will be set to NaN.
	"""
	def __init__(
		self, 
		testing=False, 
		log_transform=False, 
		quantile_filter=0.001,
		device="cpu",
):
		self.X = None
		self.X_val = None
		self.mean = None
		self.std = None
		self.testing = testing
		self.part = None
		self.log_trans = log_transform
		self.quantile_filter = quantile_filter
		self.device = device

	def fit(self, X, X_val=None, partition="Train"):
		"""
		Learn the scaling parameters for the matrix. This also does
		a basic quantile-based low-value outlier exclusion. 

		Parameters
		----------
		X : torch.tensor, 
			Input matrix; the training set
		X_val : torch.tensor, 
			Input matrix; the validation set. Note that we don't 
			necessarily use this. It's only used if `partition`
			== "Valid".
		partition : str, 
			The partition that we're trying to scale. 
			{'Train', 'Valid', 'Eval'}

		Returns
		----------
		self
		"""
		self.part = partition

		X = _check_tensor(X)
		if self.part == "Valid":
			X_val = _check_tensor(X_val)
		
		# These are most likely integrated background noise, and will
		#   lead to negative values after logging
		# if not self.testing:
		# 	X[X <= 1.0] = np.nan
		# 	if X_val is not None:
		# 		X_val[X_val <= 1.0] = np.nan
		
		# Log transform 
		if self.log_trans:
			X = torch.log(X)
			if self.part == "Valid":
				X_val = torch.log(X_val)

		# Quantile-based outlier exclusion. The try/except block
		#	is designed to catch a known torch error wherein the
		#	torch.quantile function fails for tensors above a 
		#	certain size
		# if not self.testing:
		# 	try:
		# 		qt_thresh = torch.nanquantile(X, q=self.quantile_filter)
		# 	except RuntimeError:
		# 		#print("in exception")
		# 		rng = np.random.default_rng(seed=18)
		# 		X_sub = X.clone().detach().cpu().numpy()
		# 		rng.shuffle(X_sub, axis=0)
		# 		rng.shuffle(X_sub, axis=1)
		# 		# Assumes the quants matrix has >1000 peptides
		# 		X_sub = X_sub[:1000,:]
		# 		qt_thresh = np.nanquantile(X_sub, q=self.quantile_filter)

		# 	if X_val is not None:
		# 		X_val[X_val <= qt_thresh] = np.nan

		self.X = X
		if self.part == "Valid":
			self.X_val = X_val

		# Calculate the scaling parameters on only the training set
		nonmissing = self.X[~torch.isnan(self.X)]
		self.mean = torch.mean(nonmissing)
		self.std = torch.std(nonmissing)

		return self

	def transform(self):
		"""
		Rescale the matrix according to the learned scaling
		parameters. 

		Parameters
		----------
		none

		Returns
		----------
		ret_mat: torch.Tensor,
			The rescaled matrix
		"""
		if self.part == "Train" or self.part == "Eval":
			ret_mat = self.X / self.std 
		elif self.part == "Valid":
			ret_mat = self.X_val / self.std

		return ret_mat

	def inverse_transform(self, X, X_val=None, partition="Train"):
		"""
		Reverse the rescaling of the matrix

		Parameters
		----------
		X : torch.tensor, 
			Input matrix; the training set
		X_val : torch.tensor, 
			Input matrix; the validation set. Note that we don't 
			necessarily use this. It's only used if `partition`
			== "Valid"
		partition : str, 
			The partition that we're trying to scale. 
			{'Train', 'Valid', 'Eval'}

		Returns
		----------
		torch.Tensor,
			The original, unscaled matrix
		"""
		X = _check_tensor(X)
		if self.part == "Valid":
			X_val = _check_tensor(X_val)

		if self.part == "Train" or self.part == "Eval":
			X = X * self.std 
			if self.log_trans:
				X = torch.exp(X)
			return X
		elif self.part == "Valid":
			X_val = X_val * self.std
			if self.log_trans:
				X_val = torch.exp(X_val)
			return X_val

	def fit_transform(self, X, X_val=None, partition="Train"):
		"""
		Rescale the matrix

		Parameters
		----------
		X : torch.tensor, 
			Input matrix; the training set
		X_val : torch.tensor, 
			Input matrix; the validation set. Note that we don't 
			necessarily use this. It's only used if `partition`
			== "Valid"
		partition : str, 
			The partition that we're trying to scale. 
			{'Train', 'Valid', 'Eval'}

		Returns
		----------
		torch.Tensor
			The rescaled matrix
		"""
		return self.fit(X, X_val, partition).transform()

class MinMaxScaler():
	"""
	Apply Min/Max scaling to the matrix. 

	Parameters
	----------
	none
	"""
	def __init__(self):
		self.min = None
		self.max = None

	def fit(self, X):
		"""
		Learn the scaling parameters for the matrix

		Parameters
		----------
		X : torch.tensor, 
			The input matrix

		Returns
		----------
		self
		"""
		X = _check_tensor(X)

		self.min = np.nanmin(X)
		self.max = np.nanmax(X)

		return self

	def transform(self, X):
		"""
		Rescale the matrix according to the learned scaling
		parameters. 

		Parameters
		----------
		X : torch.tensor, 
			The input matrix

		Returns
		----------
		Y: torch.Tensor, 
			The rescaled matrix
		"""
		X = _check_tensor(X)

		Y = (X - self.min) / (self.max - self.min)
		return Y

	def inverse_transform(self, X):
		"""
		Reverse the rescaling of the matrix

		Parameters
		----------
		X : torch.tensor, 
			The input matrix

		Returns
		----------
		Y : torch.Tensor,
			The original, unscaled matrix
		"""
		X = _check_tensor(X)

		Y = X * (self.max - self.min) + self.min
		return Y

	def fit_transform(self, X):
		"""
		Rescale the matrix

		Parameters
		----------
		X : torch.tensor, 
			The input matrix

		Returns
		----------
		torch.Tensor
			The rescaled matrix
		"""
		return self.fit(X).transform(X)

class LogFoldScaler():
	"""
	This one is designed specifically for the log-fold change
	TMT data. It needs to play nice with negative values. Just doing
	a very basic quantile-based scaling here. Getting rid of extreme
	values. All this does is winsorize. 
	"""
	def __init__(self):
		self.X = None
		self.X_val = None

	def fit(self, X, X_val, partition="Train"):
		"""
		Learn the scaling parameters for the matrix. This also does
		a basic quantile-based low-value outlier exclusion. 

		Parameters
		----------
		X : torch.tensor, 
			Input matrix; the training set
		X_val : torch.tensor, 
			Input matrix; the validation set. Note that we don't 
			necessarily use this. It's only used if `partition`
			== "Valid". Can be None?
		partition : str, 
			The partition that we're trying to scale. 
			{'Train', 'Valid', 'Eval'}

		Returns
		----------
		self
		"""
		X = _check_tensor(X)
		if X_val is not None: 
			X_val = _check_tensor(X_val)

		self.X = X
		self.X_val = X_val
		self.part = partition

		# Calculate the scaling parameters on only the training set
		nonmissing = self.X[~torch.isnan(self.X)]

		try:
			self.train_min = torch.nanquantile(nonmissing, 0.001)
			self.train_max = torch.nanquantile(nonmissing, 0.999)
		# This addresses a known torch/numpy issue wherein the
		#	quantile function fails for very large tensors/vectors
		# 	Here we just repeat the procedure for a subset of the 
		#	full tensor.
		except RuntimeError:
			#print("in exception")
			rng = np.random.default_rng(seed=18)
			nonmiss_np = nonmissing.detach().cpu().numpy()
			rng.shuffle(nonmiss_np)

			self.train_min = \
				torch.tensor(np.nanquantile(nonmiss_np[:8000], 0.001))
			self.train_max = \
				torch.tensor(np.nanquantile(nonmiss_np[:8000], 0.999))

		return self

	def transform(self):
		"""
		Rescale the matrix according to the learned scaling
		parameters. 
		"""
		if self.part == "Train" or self.part == "Eval":
			self.X[self.X < self.train_min] = self.train_min
			self.X[self.X > self.train_max] = self.train_max

			# Shift everything to the right
			self.X = self.X + torch.abs(self.train_min)
			return self.X

		elif self.part == "Valid":
			self.X_val[self.X_val < self.train_min] = self.train_min
			self.X_val[self.X_val > self.train_max] = self.train_max

			# Shift everything to the right
			self.X_val = self.X_val + torch.abs(self.train_min)
			return self.X_val

	def inverse_transform(self, X, X_val, partition="Train"):
		"""
		Reverse the rescaling of the matrix. Not sure how to
		implement this one. Right now I'm just returning the
		same `X` and `X_val` tensors. 

		Parameters
		----------
		X : torch.tensor, 
			Input matrix; the training set
		X_val : torch.tensor, 
			Input matrix; the validation set. Note that we don't 
			necessarily use this. It's only used if `partition`
			== "Valid"
		partition : str, 
			The partition that we're trying to scale. 
			{'Train', 'Valid', 'Eval'}

		Returns
		----------
		X or X_val : torch.tensor, 
			The (re-scaled) input matrix
		"""
		X = _check_tensor(X)
		if X_val is not None:
			X_val = _check_tensor(X_val)

		if partition == "Train" or partition == "Eval":
			# Shift back
			X = X + self.train_min
			return X
		elif partition == "Valid":
			# Shift back
			X_val = X_val + self.train_min
			return X_val

	def fit_transform(self, X, X_val, partition="Train"):
		"""
		Rescale the matrix

		Parameters
		----------
		X : torch.tensor, 
			Input matrix; the training set
		X_val : torch.tensor, 
			Input matrix; the validation set. Note that we don't 
			necessarily use this. It's only used if `partition`
			== "Valid"
		partition : str, 
			The partition that we're trying to scale. 
			{'Train', 'Valid', 'Eval'}

		Returns
		----------
		torch.Tensor
			The rescaled matrix
		"""
		return self.fit(X, X_val, partition).transform()

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
