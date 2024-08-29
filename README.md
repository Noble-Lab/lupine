`lupine`
================================

What is `lupine`?
-------------------------------------

`lupine` is a python package for missing value imputation of quantitative proteomics data with a multilayer perceptron. `lupine` is implemented in PyTorch. The only required input is a csv file containing a matrix of protein or peptide level quantifications where rows are proteins or peptides and columns are MS runs. Optional settings are described below. In our manuscript we evaluate `lupine` on TMT data, but it is also applicable to LFQ and DIA. This repository contains model weights for a pre-trained `lupine` model fit to ~1,900 clinical patient samples from CPTAC, described in our manuscript [here](https://pubs.acs.org/doi/10.1021/acs.jproteome.3c00205). Users can download this pre-trained model, append their own MS runs and fine-tune the `lupine` model to impute missing values in their own data. This will write a csv with imputed protein or peptide quantifications for the user-submitted runs. 

You can find the manuscript describing Lupine [here](https://www.biorxiv.org/content/10.1101/2024.08.26.609780v2). 

`lupine` _currently requires a GPU to train. We are working on a distilled model that can be run with reasonable runtime on a CPU._

Dependencies
------------
`lupine` is configured to run within a virtual environment, the python package dependencies for which are provided in `requirements.txt`. However, if you are having trouble with install, you may try installing the following dependencies. 

**MacOS dependencies:**
```
pip install setuptools
brew update
brew install zlib
```

**Linux dependencies:**
```
apt-get install autoconf automake gcc zlib1g libbz2 libssl
```
`lupine` is not currently installable on Windows (we're working on it). `lupine` requires python >= 3.7. 

Installation
------------
**With [conda](https://anaconda.org/anaconda/conda):**
```
conda env create --file conda_requirements.yml
conda activate lupine
pip install .
```

Usage
-----
To impute a matrix of peptide or protein quantifications:
```
lupine impute /path/to/peptide/quants/csv --args
```
The arguments are described below. The only required argument is the path to the csv. 

```
--n_prot_factors : int, the number of protein factors.          

--n_run_factors : int, the number of factors to use for the matrix factorization-based run embeddings.  

--n_layers : int, the number of hidden layers in the DNN.        

--n_nodes : int, the number of nodes in the factorization based neural network.       

--rand_seed : int, the random seed. Should probably only be set for testing and figure generation. Default is None.
     
--biased : bool, use the biased mini-batch selection procedure when creating the data loader?      

--device : str, the device to use for computation, {"cpu", "cuda"}.    
```
The above command will run `lupine` and create a results directory with a file named `lupine_recon_quants.csv`. 

Authors
--------
[Lincoln Harris](https://github.com/lincoln-harris) & [William S. Noble](https://noble.gs.washington.edu/).     
Department of Genome Sciences, University of Washington, Seattle, WA.

Contributing
------------
We welcome any bug reports, feature requests or other contributions. Please submit a well documented report on our [issue tracker](https://github.com/Noble-Lab/lupine/issues). For substantial changes please fork this repo and submit a pull request for review.

See [CONTRIBUTING.md](https://github.com/Noble-Lab/lupine/blob/main/CONTRIBUTING.md) for additional details.

You can find official releases [here](https://github.com/Noble-Lab/lupine/releases).
