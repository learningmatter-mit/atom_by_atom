# Per-site PaiNN

Description

## Installation

1. Clone the repository

```bash
git clone git@github.mit.edu:MLMat/per-site_PaiNN.git
```

2. Conda environments

```bash
conda upgrade conda
conda env create -f environment.yml
conda activate persitePainn
```

3. Install NeuralForceField (nff)

```bash
git clone https://github.com/learningmatter-mit/NeuralForceField.git
# Go to the nff directory
pip install .
# Copy nff/utils/table_data to the installed directory in conda envs python packages
```

4. Install Wandb

   Create an account [here](https://wandb.ai/home) and install the Python package:

```bash
pip install wandb
wandb login
```

5. Install

```bash
# Go to per-site_painn directory
pip install .
```

## Usage

run `main.py` with settings (e.g., `details.json` below)\

- example command line

```bash
python main.py --data data_raw/data.pkl --cache data_cache/data_cache --details details.json --savedir results
```

- example `*.json`
