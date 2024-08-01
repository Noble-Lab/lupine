"""
SCALERS

This module contains methods to scaler our inputs prior to 
imputation with our models. This module contains a single class:
	`STDScaler` - Scale the input matrix according to 
	the standard deviation of the entire matrix
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

		# Log transform 
		if self.log_trans:
			X = torch.log(X)
			if self.part == "Valid":
				X_val = torch.log(X_val)

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
