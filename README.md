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
This repository requires the following packages to run correctly:
```
```

All these packages can be installed using the [environment.yml](environment.yml) file and `conda`:
```
conda env create -f environment.yml
conda activate atombyatom
```

# Downlaoding the data from Atom by Atom Design of Metal Oxide Catalysts for the Oxygen Evolution Reaction with Machine Learning

There are several datasets for per-site properties of bulk oxides available. Download these inside the data folder using the following commands:
```
python download.py bulk_dos
```
