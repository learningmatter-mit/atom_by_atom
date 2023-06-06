This repo is under construction.

# Atom by Atom Design of Metal Oxide Catalysts for the Oxygen Evolution Reaction with Machine Learning

This software package includes all data, models, and analysis scripts necessary for reproducing "Atom by atom design of metal oxide catalysts for the oxygen evolution reaction with machine learning". More information about the data, models and analysis can be found [here](https://doi.org/10.48550/arXiv.2305.19930).


## Installation 

To clone this repo and all submodules:
```
git clone --recurse-submodules git@github.com:learningmatter-mit/atom_by_atom.git
```
or
```
git clone git@github.com:learningmatter-mit/atom_by_atom.git
git submodule update --init
```

To only update the submodules:
```
git submodule update --remote --merge
```

We recommend installing the atombyatom package using the following line:
```
pip install -e . # for developer mode
```
or 
```
pip install . 
```


## Downloading the data from Atom by Atom Design of Metal Oxide Catalysts for the Oxygen Evolution Reaction with Machine Learning

There are several datasets for per-site properties of bulk oxides available. You can download the datasets using the following commands: 
```
atombyatom download bulk_dos
```

The datasets will be downloaded inside the data folder as data/bulk_dos.json etc.

## Running the per-site cgcnn and per-site painn codes

Per-site CGCNN and Per-site PAINN can be trained/tested on the downloaded datasets by running the following line:
```
atombyatom run model_name --dataset dataset_name
```

For example, to run per-site_cgcnn on the bulk_dos data, you would run the following line:
```
atombyatom run per-site_cgcnn --dataset bulk_dos
```

The results of running the model (including the train/val/test results, and a checkpoint of the best model) are stored in the atombyatom/results/model_name/dataset_name directory. the README.md files inside of the per-site_cgcnn and per-site_painn submodules for more details about these two models.  
