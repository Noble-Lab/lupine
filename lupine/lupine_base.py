"""
LUPINE_BASE
1.16.23

The abstract base class for a model that randomly generates protein 
and run embeddings, then refines them with stochastic gradient descent.
Missing values are imputed by taking the concatenation of the 
corresponding protein and run factors and feeding them through a deep
neural net. Configured for protein-level imputation, for the quants
matrices from the University of Michigan's data processing pipeline
for CPTAC. 
"""
import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import ranksums

# Import my classes and modules
from data_loaders import FactorizationDataset
from scalers import STDScaler
from utils import plot_partition_distributions

# Plotting templates
sns.set(context="talk", style="ticks") 
sns.set_palette("tab10")

torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True

# Got sick of looking at the PyTorch nested tensors warning
#	but generally this is a bad idea
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class LupineBase(torch.nn.Module):
	"""
	The base class for the matrix factorization-based imputation
	model. Protein and run embeddings are randomly initialized,
	then refined through SGD. The forward pass is specified in the
	`Lupine` class, but this class does the rest of the
	heavy lifting. 

	Parameters
	----------
	n_prots : int, 
		The number of proteins in the quants matrix
	n_runs : int, 
		The number of runs in the protein quants matrix
	n_prot_factors : int, optional,
		The number of protein factors
	n_run_factors : int, optional,
		The number of factors to use for the matrix factorization-
		based run embeddings
	n_layers : int, optional, 
		The number of hidden layers in the DNN.
	n_nodes : int, optional,
		The number of nodes in the factorization based 
		neural network.
	learning_rate : float, optional,
		The learning rate for the model's Adam optimizer
	batch_size : int, optional,
		The number of matrix X_ijs to assign to each mini-batch
	tolerance : float, optional,
		The tolerance criteria for early stopping, according to the
		standard early stopping criteria
	max_epochs : int, optional,
		The maximum number of training epochs for the model
	patience : int, optional
		The number of training epochs to wait before stopping if
		it seems like the model has converged
	q_filt : float, optional,
		The quantile of low values to set to NaN when scaling the
		data
	rand_seed : int, optional,
		The random seed. Should probably only be set for testing and
		figure generation. Default is `None`.
	testing : bool, optional,
		Is the model being run in testing mode? If yes, random seeds
		will be set manually
	biased : bool, optional,
		Use the biased mini-batch selection procedure when creating
		the data loader? 
	device : str, optional,
		The device to use for computation. {"cpu", "cuda"}
	"""
	def __init__(
		self, 
		n_prots,
		n_runs, 
		n_prot_factors=128,
		n_run_factors=128,
		n_layers=2,
		n_nodes=128,
		learning_rate=0.01,
		batch_size=128,
		tolerance=0.001,
		max_epochs=512,
		patience=10,
		q_filt=0.001,
		rand_seed=None,
		testing=False,
		biased=False,
		device="cpu",
	):
		super().__init__()

		self.device = device
		print("Loading tensors on: ", self.device)

		self.n_prot_factors = n_prot_factors
		self.n_run_factors = n_run_factors
		self.n_layers = n_layers
		self.n_nodes = n_nodes
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.tolerance = tolerance
		self.max_epochs = max_epochs
		self.patience = patience
		self.rand_seed = rand_seed
		self.testing = testing
		self.biased = biased 
		
		# This should really only be specified for testing and 
		#   figure generation
		if self.rand_seed:
			torch.manual_seed(self.rand_seed)

		# For writing the model state to disk
		self.MODELPATH = "./OPT_MODEL_INTERNAL.pt"

		# Need to remove the previously saved model before training
		#    a new one. FIXME: is there a better way to do this?
		try:
			os.remove(self.MODELPATH)
		except FileNotFoundError:
			pass

		# Init the run factors
		self.run_factors = torch.nn.Parameter(
								torch.randn(
									self.n_run_factors, 
									n_runs, 
									requires_grad=True,
									device=self.device,
								)
		)
		# Randomly init the protein factors
		self.prot_factors = torch.nn.Parameter(
								torch.randn(
									n_prots,
									self.n_prot_factors,
									requires_grad=True,
									device=self.device,
								)
		)
		# Init the DNN, with variable number of hidden layers
		dnn = []
		# Add the input layer
		dnn.append( 
			torch.nn.Linear(
				self.n_prot_factors + self.n_run_factors, 
				self.n_nodes,
				device=self.device)
		)
		dnn.append(torch.nn.LeakyReLU(0.1))

		# Add the hidden layers
		for k in range(0,self.n_layers):
			dnn.append(
				torch.nn.Linear(
					self.n_nodes, 
					self.n_nodes, 
					device=self.device
				)
			)
			dnn.append(torch.nn.LeakyReLU(0.1))

		# Add the output layer
		dnn.append(
			torch.nn.Linear(self.n_nodes, 1, device=self.device)
		)
		# Add the final transformation
		#dnn.append(torch.nn.Softplus())
		dnn.append(torch.nn.LeakyReLU(0.1))
		# Convert to sequential
		self.dnn = torch.nn.Sequential(*dnn)

		# Init the loss function 
		self.loss_fn = torch.nn.MSELoss(reduction="mean")
		#self.loss_fn = self._nrmse_loss

		# Init the scaler
		self.scaler = STDScaler(testing=False, log_transform=False)
		# Init the optimizer
		self.optimizer = torch.optim.Adam(
							self.parameters(), 
							lr=self.learning_rate,
		)

		self._history = []
		self._stopping_criteria = None

		total_params = sum(p.numel() for p in self.parameters())
		print(f"Number of parameters: {total_params}")

	def fit(self, X_mat, X_val_mat=None):
		"""
		Fit the model. Scales the training and validation sets,
		gets data loaders, trains the model, where for each 
		training epoch we get new protein embeddings, take the
		scalar product and calculate loss and gradients. The 
		validation set is optional. 

		Parameters
		----------
		X : np.ndarray, 
			The training protein quants matrix
		X_val : np.ndarray, optional,
			The validation protein quants matrix

		Returns
		----------
		self
		"""
		validate=False
		if X_val_mat is not None:
			validate = True

		# Scale the training and validation sets
		X = self.scaler.fit_transform(
							X_mat, 
							X_val_mat, 
							partition="Train", 
		)
		if validate:
			X_val_mat = self.scaler.fit_transform(
							X_mat, 
							X_val_mat, 
							partition="Valid", 
			)
		# Generate the initial data loaders
		train_loader = FactorizationDataset(
								X,
								X_val_mat, 
								partition="Train",
								batch_size=self.batch_size, 
								biased=self.biased,
								shuffle=True, 
								missing=False,
								testing=self.testing,
								anchor=0.2,
								iters=3,
								std=0.4,
								b_prob=0.8,
								rand_seed=self.rand_seed,
		)
		if validate:
			val_loader = FactorizationDataset(
								X,
								X_val_mat, 
								partition="Valid",
								batch_size=self.batch_size, 
								biased=False,
								shuffle=True, 
								missing=False,
								testing=self.testing,
								rand_seed=self.rand_seed,
			)
		# Get the mini-batches -- biased selection for the train
		#   loader, non-biased selection for the validation loader
		if self.biased:
			train_loader.get_biased_loader()
		else:
			train_loader.get_standard_loader()

		if validate:
			val_loader.get_standard_loader()

		# sanity checking the distributions of the partitions
		# plot_partition_distributions(
		# 		X.detach().cpu().numpy(), 
		# 		train_loader, val_loader,
		# 		#outstr="test", save_fig=True,
		# )

		# Evaluate the model prior to training
		train_loss = self._evaluate(train_loader, 0, "Train")
		if validate:	
			val_loss = self._evaluate(val_loader, 0, "Validation")

		best_loss = np.inf
		stopping_counter = 0

		# Train an epoch
		for epoch in tqdm(range(1, self.max_epochs+1), unit="epoch"):
			self.train()
			# Train a mini-batch
			for locs, target in train_loader:
				target = target.type_as(self.run_factors)

				# Reset the gradients
				self.optimizer.zero_grad()

				# Get the predictions 
				preds = self(locs)

				# Get the training loss
				train_loss = self.loss_fn(preds, target)

				# Compute new gradients and take the SGD step
				train_loss.backward()
				self.optimizer.step()

			# Get train and validation loss, after each epoch
			train_loss = self._evaluate(
						train_loader, epoch, "Train")
			if validate:
				val_loss = self._evaluate(
						val_loader, epoch, "Validation")

			# Checkpoint, if the curr validation loss is lower than 
			#   the lowest yet recorded validation loss
			if validate:
				curr_loss = val_loss 
			else:
				curr_loss = train_loss

			if curr_loss < best_loss:
				torch.save(self, self.MODELPATH)
				best_loss = curr_loss

			# Evaluate early stopping:
				# has validation loss plateaued? 
			if self.tolerance > 0 and epoch > 16:
				tol = torch.abs((best_loss - curr_loss) \
											/ best_loss)
				loss_ratio = curr_loss / best_loss

				if tol < self.tolerance:
					stopping_counter += 1
				else:
					stopping_counter = 0

				if stopping_counter == self.patience:
					self._stopping_criteria = "standard"
					break

			# Evaluate early stopping: 
				# is validation loss going back up?
			if validate:
				if self.tolerance > 0 and epoch > 16:
					window2 = np.array(
							self.history["Validation"][-5:])
					window1 = np.array(
							self.history["Validation"][-15:-10])

					wilcoxon_p = ranksums(
							window2, window1, 
							alternative="greater")[1]

					if wilcoxon_p < 0.05:
						self._stopping_criteria = "wilcoxon"
						break

			# Generate new data loaders at the end of every epoch
			train_loader = FactorizationDataset(
									X,
									X_val_mat, 
									partition="Train",
									batch_size=self.batch_size, 
									biased=self.biased,
									shuffle=True, 
									missing=False,
									testing=self.testing,
									anchor=0.2,
									iters=3,
									std=0.4,
									b_prob=0.8,
									rand_seed=self.rand_seed,
			)
			if validate:
				val_loader = FactorizationDataset(
									X,
									X_val_mat, 
									partition="Valid",
									batch_size=self.batch_size, 
									biased=False,
									shuffle=True, 
									missing=False,
									testing=self.testing,
									rand_seed=self.rand_seed,
				)
			# Get the next set of mini-batches
			if self.biased:
				train_loader.get_biased_loader()
			else:
				train_loader.get_standard_loader()

			if validate:
				val_loader.get_standard_loader()

		return self

	def _evaluate(self, loader, epoch, set_name):
		"""
		Evaluate model progress, during training. 
		Private function. Note that this calculates the loss across
		the *entire* training or validation set and not the current 
		mini-batch. 
		
		Parameters
		----------
		loader : torch.DataLoader,
			The data loader to use for evaluation 
		epoch : int, 
			The current training epoch
		set_name :  str, 
			The name to use in the history: {"Train", "Validation"}

		Returns
		----------
		curr_loss : tensor, 
			The {training, validation} loss for the current epoch
		"""
		self.eval()
		
		with torch.no_grad():
			res = [(self(l), t) for l, t in loader]

			pred, target = list(zip(*res))
			
			pred = torch.cat(pred, dim=0)
			target = torch.cat(target, dim=0)
			target = target.type_as(self.run_factors)

			curr_loss = self.loss_fn(pred, target)

			try:
				self._history[epoch][set_name] = curr_loss.item()
			except IndexError: # the first one
				self._history.append(
					{"epoch": epoch, set_name: curr_loss.item()})

		return curr_loss

	def _nrmse_loss(self, x_vals, y_vals):
		"""
		Get the (normalized) Root Mean Squared Error loss between 
		two tensors. One question is how to actually do the 
		normalization: could be standard deviation, could be mean, 
		could be (max-min).
		
		Parameters
		----------
		x_vals, y_vals : torch.tensor, 
			The observed and expected values, respectively

		Returns
		----------
		nrmse_loss : torch.tensor 
			The normalized root mean squared error loss
		"""
		# Exclude NaNs
		x_rav = x_vals.ravel()
		y_rav = y_vals.ravel()
		missing = torch.isnan(x_rav) | torch.isnan(y_rav)

		# Get the MSE
		mse = torch.sum((x_rav[~missing] - y_rav[~missing])**2) \
											/ torch.sum(~missing)
		# Get the RMSE
		rmse = torch.sqrt(mse)

		# Normalize by the standard deviation of the expected values
		#	How to do the normalization?
		y_std = torch.std(y_rav[~missing])
		y_mean = torch.mean(y_rav[~missing])
		y_diff = \
			torch.max(y_rav[~missing]) - torch.min(y_rav[~missing])
		nrmse = rmse / y_std

		return nrmse

	def transform(self, X, X_val):
		"""
		Impute missing values with the learned model. I think 
		right now this is set up to impute both the validation and
		test sets. This is fine, so long as we're only calculating
		the reconstruction error on the test set. 

		Parameters
		----------
		X : array-like, 
			The matrix to factorize. The training set. These values 
			will not be transformed
		X_val : array-like, 
			The validation set. These values will be transformed,
			as will the test set

		Returns
		----------
		np.ndarray, 
			The imputed matrix. Only the missing values 
			in X are imputed
		"""
		self.eval()

		X = _check_tensor(X)

		# Get the scaler
		X = self.scaler.fit_transform(X, X_val, partition="Eval")

		# Init the data loader. Here `missing` should be set to True
		#	because we actually want a list of the missing X_ijs
		# 	for the trained model to reconstruct. Here we also need
		#	to do the STANDARD (i.e. NON-BIASED) mini-batch selection
		#	procedure.
		loader = FactorizationDataset(
								X, 
								X_val=None,
								partition="Train",
								batch_size=self.batch_size, 
								biased=False,
								shuffle=True, 
								missing=True,
								testing=self.testing,
								rand_seed=self.rand_seed,
		)
		# Get the mini-batches -- standard data loader
		loader.get_standard_loader()

		with torch.no_grad():
			for locs, _ in loader:
				target = X[tuple(locs.T)]
				X[tuple(locs.T)] = self(locs).type_as(target)

		# Inverse scale. This should use the train set scaling factor
		X = self.scaler.inverse_transform(X, X_val, "Eval")

		return X.detach().cpu().numpy()

	def full_transform(self, X):
		"""
		Impute the entire matrix with the learned model, 
		including the training set. For diagnostic purposes
		only.

		Parameters
		----------
		X : array-like, 
			The matrix to factorize

		Returns
		----------
		X_imp : np.ndarray, 
			The entirely reconstructed matrix. The training set
			and the initially missing values are transformed. 
		"""
		raise NotImplementedError

	def fit_transform(self, X, X_val=None):
		"""
		Fit the model, then impute missing values with the
		learned model

		Parameters
		----------
		X_mat : np.ndarray, 
			The training protein quants matrix
		X_val_mat : np.ndarray, optional,
			The validation protein quants matrix

		Returns
		----------
		X_imp : np.ndarray, 
			The imputed matrix. Only the missing values 
			in X are imputed
		"""
		return self.fit(X, X_val).transform(X, X_val)

	def plot_loss_curves(self):
		""" 
		Generate model loss vs training epoch plot. For both training
		and validation sets. A basic sanity check method. Note that 
		the scale of the y axis will reflect the scaled values. 

		Parameters
		----------
		self
		"""
		plt.figure()
		plt.plot(list(self.history.epoch[1:]), 
			list(self.history["Train"][1:]), 
			label="Training loss")
		plt.plot(list(self.history.epoch[1:]), 
			list(self.history["Validation"][1:]), 
			label="Validation loss")

		plt.ylim(ymin=0)

		plt.legend()
		plt.xlabel("Epochs")
		plt.ylabel("MSE")

		plt.show()
		plt.close()

		return

	@property
	def history(self):
		""" Training and validation loss for every iter """
		return pd.DataFrame(self._history)

	@property
	def stopping_criteria(self):
		""" 
		Returns the stopping criteria that was ended model
		training. {"standard", "wilcoxon"}. "None" indicates that 
		early stopping was not triggered. 
		"""
		return self._stopping_criteria

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
