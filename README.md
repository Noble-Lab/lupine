What is Lupine?
-------------------------------------

Lupine is a python package for imputing missing values in mass spectrometry (MS) proteomics data with a multilayer perceptron. You can find the manuscript describing Lupine [here](https://www.biorxiv.org/content/10.1101/2024.08.26.609780v2). Lupine requires a csv file containing protein-level quantifications where rows are proteins, columns are MS runs (or in the case of TMT, demultiplexed TMT samples), and values are protein intensities. Lupine is agnostic to the tools/methods used to generate protein quants from raw files; the model is only concerned with protein quants. 

Lupine is unique in that it learns from many MS experiments and runs to impute missing protein quantification values. This repository includes a zip file containing a "joint quantifications" matrix consisting of demultiplexed TMT runs from 10 experiments that were part of the [CPTAC](https://pdc.cancer.gov/pdc/cptac-pancancer) project. You will see the best performance if you attach your MS runs to this joint quantifications matrix and fit Lupine to the merged quantifications matrix. The `join` module is designed to help with this task. However, Lupine may also be fit directly to your MS runs. For this you would proceed directly to the `impute` module. We don't necessarily recommend this practice, as without enough training data, the Lupine model is likely to overfit. 

Lupine is an ensemble of individual models each trained with different combinations of hyperparameters. These hyperparameters are automatically selected by the model. Accordingly, Lupine is quite slow and requires a GPU to train. We are currently working on a faster and CPU-enabled version of the method. Lupine is implemented in PyTorch. 

Dependencies
------------
Lupine is configured to run within a virtual environment, the python package dependencies for which are provided in `requirements.txt`. However, if you are having trouble with install, you may try installing the following dependencies. 

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
Lupine is not currently installable on Windows. Lupine requires python >= 3.7. 

Installation
------------
**With [conda](https://anaconda.org/anaconda/conda):**
```
conda env create --file conda_requirements.yml
conda activate lupine
pip install .
```

The next step is to navigate to the latest [release](https://github.com/Noble-Lab/lupine/releases) and download `data.zip` under the *Assets* tab. This should be downloaded to the Lupine project directory. 

Usage
-----
Lupine is comprised of three modules: 
```
convert : Convert between ENSG or HGNC protein identifiers to ENSPs. 
join : 	Add your MS runs to Lupine's training matrix, prior to Lupine imputation. 
impute : Impute missing values in a protein quantification matrix. 
```

The input to the method is a proteins-by-samples matrix where rows are proteins and columns are MS runs or samples. As an example, we have included the file `CCLE_quants_tester.csv`, which is a matrix of protein quantifications from the [Cancer Cell Line Encyclopedia](https://gygi.hms.harvard.edu/publications/ccle.html) project: 

<p align="center">
    <img src="https://github.com/Noble-Lab/lupine/blob/main/docs/ccle_quants_tester_ss.png" width="500">
</p>

Note that all extraneous metadata columns have been removed: the remaining columns consist solely of protein intensities for each sample. Each protein is specified by a unique Ensemble [ENSP](https://useast.ensembl.org/info/genome/stable_ids/index.html) ID; this is required by the method. If this is not the case for your data, the `convert` module can help convert between ENSG, [HGNC](https://www.genenames.org/) and ENSP IDs. 

Once you have your data in the correct format, the `join` module may be used to add your MS runs to Lupine's training dataset. The options for this step are described below: 
```
--csv : Path to the CSV file containing the MS runs to impute
--log_transform: Log transform the MS runs? {True, False}
```
If your MS runs have not previously been log transformed, you should set the `log_transform` parameter to True. 

The final step is to impute your MS runs with the `impute` module. The options for this step are described below: 
```
--csv : Path to the merged CSV file
--outpath : Output directory
--n_models : The number of models to fit. (Default: 8)
--biased : Biased batch selection? (Default: True)
--device : The device to load the model on. (Default: cuda)
--mode : The model run mode. (Default: run)
```
Typically the last four parameters do not need to be specified. This step will produce a file within the specified directory named `lupine_recon_quants.csv`. 

Authors
--------
[Lincoln Harris](https://github.com/lincoln-harris) & [William S. Noble](https://noble.gs.washington.edu/).     
Department of Genome Sciences, University of Washington, Seattle, WA.

Contributing
------------
We welcome any bug reports, feature requests or other contributions. Please submit a well documented report on our [issue tracker](https://github.com/Noble-Lab/lupine/issues). For substantial changes please fork this repo and submit a pull request for review.

See [CONTRIBUTING.md](https://github.com/Noble-Lab/lupine/blob/main/CONTRIBUTING.md) for additional details.

You can find official releases [here](https://github.com/Noble-Lab/lupine/releases).
