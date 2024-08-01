lupine
================================

[![image](https://img.shields.io/travis/%7B%7B%20cookiecutter.github_username%20%7D%7D/%7B%7B%20cookiecutter.repo_name%20%7D%7D.svg)](https://travis-ci.org/%7B%7B%20cookiecutter.github_username%20%7D%7D/%7B%7B%20cookiecutter.repo_name%20%7D%7D)

[![codecov](https://codecov.io/gh/%7B%7B%20cookiecutter.github_username%20%7D%7D/%7B%7B%20cookiecutter.repo_name%20%7D%7D/branch/master/graph/badge.svg)](https://codecov.io/gh/%7B%7B%20cookiecutter.github_username%20%7D%7D/%7B%7B%20cookiecutter.repo_name%20%7D%7D)

[![image](https://img.shields.io/pypi/v/%7B%7B%20cookiecutter.repo_name%20%7D%7D.svg)](https://pypi.python.org/pypi/%7B%7B%20cookiecutter.repo_name%20%7D%7D)


What is lupine?
-------------------------------------

TMT proteomics imputation with deep matrix factorization

-   Free software: MIT license
-   Documentation: <https://github.com/Noble-Lab/lupine/>

Installation
------------

To install this code, clone this github repository and use pip to install

```
git clone https://github.com/Noble-Lab/lupine.git
cd lupine 

pip install . 
```

Usage
-----
`n_prots` : _int_,    
    The number of proteins in the quants matrix. Required.     

`n_runs` : _int_,    
    The number of runs in the protein quants matrix. Required.       

`n_prot_factors` : _int_,    
    The number of protein factors. Optional.          

`n_run_factors` : _int_,   
    The number of factors to use for the matrix factorization-based run embeddings. Optional.        

`n_layers` : _int_,     
    The number of hidden layers in the DNN. Optional.        

`n_nodes` : _int_,      
    The number of nodes in the factorization based neural network. Optional.             

`rand_seed` : _int_,           
    The random seed. Should probably only be set for testing and figure generation. Default is `None`. Optional.           
`biased` : _bool_,           
    Use the biased mini-batch selection procedure when creating the data loader? Optional.      

`device` : _str_,    
    The device to use for computation, {"cpu", "cuda"}. Optional.     

Features
--------

TODO: add this section. 

