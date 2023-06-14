import pickle as pkl
import matplotlib.pyplot as plt
import argparse
from atombyatom.analysis.utils import plot_hexbin
import numpy as np

# function for flattening list of lists
def flatten(l):
    return [item for sublist in l for item in sublist]

# create parser
parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', type=str, default='results/')
parser.add_argument('--model', type=str, default='per-site_crabnet')
parser.add_argument('--dataset', type=str, default='bulk_dos')
parser.add_argument('--site_prop', type=str, default='bandcenter')

# parse arguments
args = parser.parse_args()

# load results
test_ids = pkl.load(open(args.results_dir + '/test_ids.pkl', 'rb'))
test_targets = pkl.load(open(args.results_dir + '/test_targs.pkl', 'rb'))
test_preds = pkl.load(open(args.results_dir + '/test_preds.pkl', 'rb'))

if args.model == 'per-site_crabnet':
    
    test_targets = flatten(test_targets)
    test_preds = flatten(test_preds)

elif args.model == 'per-site_cgcnn':

    site_targs = []
    site_preds = []

    for index in range(len(test_ids)):

        id_ = test_ids[index]
    
        targ = test_targets[index][:,1].numpy()
        pred = test_preds[index][:,1].numpy()

        site_targs.append(targ)
        site_preds.append(pred)

    test_targets = flatten(site_targs)
    test_preds = flatten(site_preds)

else:
    raise NotImplementedError

indexes = np.where(~np.isnan(np.array(test_targets)))[0]

fig, ax = plt.subplots(figsize=(5,5))
_, _, ax, _ = plot_hexbin(np.array(test_targets)[indexes], np.array(test_preds)[indexes], fig, ax, bins='log', cmap='gray_r')

plt.xlabel('calculated site band center (eV)')
plt.ylabel('predicted site band center (eV)')
plt.show()