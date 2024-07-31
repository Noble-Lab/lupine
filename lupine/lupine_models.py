"""
LUPINE_MODELS
1.16.23

This module contains the high-level implementation for a single
PyTorch model for imputing protein-level quants using deep matrix
factorization. 

Missing values are imputed by taking the concatenation of the 
corresponding protein and run factors and feeding them through a deep
neural net. For the quants matrices from the University of Michigan's 
data processing pipeline for CPTAC. 
"""
from lupine_base import LupineBase
import torch

class Lupine(LupineBase):
	"""
	A deep neural network-based matrix factorization imputation 
	model. Protein and run embeddings are randomly initialized,
	then refined through SGD. The forward pass is specified here,
	but the `LupineBase` class does all of the heavy lifting.

	Parameters
	----------
	n_prots : int, 
		The number of proteins in the quants matrix
	n_runs : int, 
		The number of runs in the protein quants matrix
	n_prot_factors : int, optional,
		The number of factors to embed each protein with
	n_run_factors : int, optional,
		The number of factors to use for the matrix factorization-
		based run embeddings
	n_layers : int, optional, 
		The number of hidden layers in the DNN.
	n_nodes : int, optional,
		The number of nodes in the factorization based neural 
		network. 
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
		figure generation. 
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
		biased=True,
		device="cpu",
	):
		super().__init__(
			n_prots=n_prots,
			n_runs=n_runs, 
			n_prot_factors=n_prot_factors,
			n_run_factors=n_run_factors,
			n_layers=n_layers,
			n_nodes=n_nodes,
			learning_rate=learning_rate,
			batch_size=batch_size,
			tolerance=tolerance,
			max_epochs=max_epochs,
			patience=patience,
			q_filt=q_filt,
			rand_seed=rand_seed,
			testing=testing,
			biased=biased,
			device=device,
		)

	def forward(self, locs):
		"""
		The forward pass of the model. For the current mini-batch,
		get the corresponding protein and run factors, concat
		and feed through the neural network to get a prediction. 

		Parameters
		----------
		locs : torch.tensor, of shape (batch_size, 2)
			The matrix indicies corresponding to the current
			mini-batch

		Returns
		----------
		torch.tensor of shape (batch_size, )
			The predicted values for the specified indices
		"""
		# Get the protein factors corresponding to the current batch
		prot_emb = self.prot_factors[locs[:,0],:]

		# Grab the 1th indicies of locs. Use these
		#    to index the cols of `self.run_factors`
		col_factors = self.run_factors[:,locs[:,1]].T

		# Concat and feed to the NN
		factors = torch.cat([prot_emb, col_factors], axis=1)

		preds = self.dnn(factors)
		preds = preds.squeeze(dim=1)
		
		return preds
