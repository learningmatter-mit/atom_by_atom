This repo is under construction.

# Atom by Atom Design of Metal Oxide Catalysts for the Oxygen Evolution Reaction with Machine Learning

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

## Installation 

We recommend installing the atombyatom package using the following line:
```
pip install -e . # for developer mode
```
or 
```
pip install . 
```


# Downloading the data from Atom by Atom Design of Metal Oxide Catalysts for the Oxygen Evolution Reaction with Machine Learning

There are several datasets for per-site properties of bulk oxides available. You can download the datasets using the following commands: 
```
atombyatom download bulk_dos
```

The datasets will be downloaded inside the data folder as data/bulk_dos.json etc.

# Running the per-site cgcnn and per-site painn codes

Per-site CGCNN and Per-site PAINN can be trained/tested on the downloaded datasets by running the following line inside of the per-site_cgcnn and per-site_painn folders, respectively:
```
python main.py --data path_to_data --dataset_cache 

See the README.md files inside of the per-site_cgcnn and per-site_painn submodules for more details. 
