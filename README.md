This repo is under construction.

# Atom by Atom Design of Metal Oxide Catalysts for the Oxygen Evolution Reaction with Machine Learning

This software package includes all data, models, and analysis scripts necessary for reproducing "Atom by atom design of metal oxide catalysts for the oxygen evolution reaction with machine learning". More information about the data, models and analysis can be found [here](https://doi.org/10.48550/arXiv.2305.19930).


## Installation 

To clone this repo and all submodules:
```
git clone git@github.com:learningmatter-mit/atom_by_atom.git
```

We recommend installing the atombyatom package using the following line:
```
pip install -e . # for developer mode
```
or 
```
pip install . 
```

## Setting up the environment
An conda environment for this repo is provided in the environment.yml file. To build and activate this environment:
```
conda env create -f environment.yml
conda activate atombyatom
```
 

## Downloading the data from Atom by Atom Design of Metal Oxide Catalysts for the Oxygen Evolution Reaction with Machine Learning

There are several datasets for per-site properties of bulk oxides available. You can download the datasets using the following commands: 
```
atombyatom download --dataset bulk_dos    # dataset for bandcenters/bandwidths/magnetic moments
atombyatom download --dataset bulk_bader  # dataset for bader charges
atombyatom download --dataset bulk_phonon # dataset for phonon bandcenters
```

Where the bulk_bader and bulk_phonon datasets are downloaded and modified from [the Materials Project](https://doi.org/10.1063/1.4812323). The datasets will be downloaded inside the data folder as data/bulk_dos.json etc.

## Running the per-site cgcnn, per-site painn, and per-site crabnet codes

Per-site models can be trained/tested on the downloaded datasets by running the following line:
```
atombyatom run --model model_name --dataset dataset_name
```

For example, to run per-site_cgcnn on the bulk_dos data, you would run the following line:
```
atombyatom run --model per-site_cgcnn --dataset bulk_dos
```

This command assumes you have already downloaded the dataset of interest, and that this dataset is available in the appropriate data folder and with the appropriate name (see above). The results of running the model (including the train/val/test results, and a checkpoint of the best model) are stored in the atombyatom/results/model_name/dataset_name directory. the README.md files inside of the per-site_cgcnn, per-site_painn and per-site_crabnet submodules for more details about these models. 

## Analyzing the model results

After running the models, it is possible to plot results from the test set using the following line:
```
atombyatom analyze --model model_name --dataset dataset_name --site_prop site_prop
```
This command assumes that you have already downloaded the dataset of interest and run the model of interest on this dataset. A parity plot will be generated of calculated vs. predicted site properties. Please note that this analysis script is very basic and will include all sites, does not differentiate between surface atoms and bulk atoms, nor between oxygens and metals.
