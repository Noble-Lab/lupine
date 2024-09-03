"""
LUPINE_BASE
8.4.24

The abstract base class for a model that randomly generates protein 
and run embeddings, then refines them with stochastic gradient 
descent. Missing values are imputed by taking the concatenation of 
the corresponding protein and run factors and feeding them through 
a deep neural net. Configured for protein-level imputation. 
"""
import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm
from scipy.stats import ranksums

# Import our classes and modules
from .data_loaders import FactorizationDataset
from .scalers import STDScaler

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
		The maximum number of training epochs for the model. 
		Default 42. 
	patience : int, optional
		The number of training epochs to wait before stopping if
		it seems like the model has converged
	rand_seed : int, optional,
		The random seed. Default is `None`.
	testing : bool, optional. 
		Default is "False". Is the model being run in testing mode?
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
		max_epochs=42,
		patience=10,
		rand_seed=None,
		testing=False,
		biased=False,
		device="cpu",
	):
		super().__init__()

		self.device = device

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
		
		if self.rand_seed:
			torch.manual_seed(self.rand_seed)

		# For writing the model state to disk
		self.MODELPATH = "scratch/OPT_MODEL_INTERNAL.pt"
		# try:
		# 	os.remove(self.MODELPATH)
		# except FileNotFoundError:
		# 	pass

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
		dnn.append(torch.nn.LeakyReLU(0.1))
		# Convert to sequential
		self.dnn = torch.nn.Sequential(*dnn)

		# Init the loss function 
		self.loss_fn = torch.nn.MSELoss(reduction="mean")

		# Init the scaler
		self.scaler = STDScaler(testing=False, log_transform=False)
		# Init the optimizer
		self.optimizer = torch.optim.Adam(
							self.parameters(), 
							lr=self.learning_rate,
		)

		self._history = []
		self._stopping_criteria = None

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
				# Has validation loss plateaued? 
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
				# Is validation loss going back up?
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
		Evaluate model progress, during training. Private function.
		Note that this calculates the loss across the *entire* 
		training or validation set and not the current mini-batch. 
		
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
