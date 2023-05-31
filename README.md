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

Data for training per-site CGCNN can be found in the data folder. 
