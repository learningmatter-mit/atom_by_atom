import pickle as pkl
import matplotlib.pyplot as plt
import argparse
from atombyatom.analysis.utils import plot_hexbin
import numpy as np
import json

from pymatgen.core.structure import Structure

# function for flattening list of lists
def flatten(l):
    return [item for sublist in l for item in sublist]

# create parser
parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', type=str, default='results/')
parser.add_argument('--data', type=str)  
parser.add_argument('--site_prop', type=str, default='bandcenter')

# parse arguments
args = parser.parse_args()

# load results
test_ids = pkl.load(open(args.results_dir + '/' + args.site_prop +  '_test_ids.pkl', 'rb'))
test_targets = pkl.load(open(args.results_dir + '/' + args.site_prop +  '_test_targs.pkl', 'rb'))
test_preds = pkl.load(open(args.results_dir + '/' + args.site_prop +  '_test_preds.pkl', 'rb'))

test_targets = flatten(test_targets)
test_preds = flatten(test_preds)


indexes = np.where(~np.isnan(np.array(test_targets)))[0]
  
fig, ax = plt.subplots(figsize=(5,5))
_, _, ax, _ = plot_hexbin(np.array(test_targets)[indexes], np.array(test_preds)[indexes], fig, ax, bins='log', cmap='gray_r')

# get units
if args.site_prop == 'bandcenter' or args.site_prop == 'bandwidth':
    units = 'eV'
elif args.site_prop == 'magmom':
    units = 'mu_B'

# get labels 
xlabel = 'calculated ' + args.site_prop + ' (' + units + ')'
ylabel = 'predicted ' + args.site_prop + ' (' + units + ')'

plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.show()
