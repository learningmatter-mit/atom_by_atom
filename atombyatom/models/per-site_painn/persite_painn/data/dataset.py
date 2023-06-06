import torch

import numbers
import numpy as np
import copy
from copy import deepcopy
from sklearn.utils import shuffle as skshuffle

from ase import Atoms
from ase.neighborlist import neighbor_list
from torch.utils.data import Dataset as TorchDataset

# TODO migrate from nff
from nff.data.parallel import (
    featurize_parallel,
    NUM_PROCS,
    add_e3fp_parallel,
    add_kj_ji_parallel,
    add_bond_idx_parallel,
)
from nff.data.features import ATOM_FEAT_TYPES, BOND_FEAT_TYPES
from nff.data.features import add_morgan as external_morgan
from nff.data.features import featurize_rdkit as external_rdkit
from nff.data.graphs import (
    reconstruct_atoms,
    get_neighbor_list,
    generate_subgraphs,
    DISTANCETHRESHOLDICT_Z,
    get_angle_list,
    add_ji_kj,
    make_dset_directed,
)


class Dataset(TorchDataset):
    """Dataset to deal with NFF calculations.

    Attributes:
        props (dict of lists): dictionary, where each key is the name of a property and
            each value is a list. The element of each list is the properties of a single
            geometry, whose coordinates are given by
            `nxyz`.

            Keys are the name of the property and values are the properties. Each value
            is given by `props[idx][key]`. The only mandatory key is 'nxyz'. If inputting
            energies, forces or hessians of different electronic states, the quantities
            should be distinguished with a "_n" suffix, where n = 0, 1, 2, ...
            Whatever name is given to the energy of state n, the corresponding force name
            must be the exact same name, but with "energy" replaced by "force".

            Example:

                props = {
                    'nxyz': [np.array([[1, 0, 0, 0], [1, 1.1, 0, 0]]),
                             np.array([[1, 3, 0, 0], [1, 1.1, 5, 0]])],
                    'lattaice': [[np.array([1,0,0])],[np.array([0,1,0])],[np.array([0,0,1])]]
                    'site_prop': [[]]
                    'target': [[]]
                }

            Periodic boundary conditions must be specified through the 'offset' key in
                props. Once the neighborlist is created, distances between
                atoms are computed by subtracting their xyz coordinates
                and adding to the offset vector. This ensures images
                of atoms outside of the unit cell have different
                distances when compared to atoms inside of the unit cell.
                This also bypasses the need for a reindexing.

        units (str): units of the energies, forces etc.

    """

    def __init__(
        self,
        props,
        check_props=True,
        do_copy=True,
    ):
        """Constructor for Dataset class.

        Args:
            props (dictionary of lists): dictionary containing the
                properties of the system. Each key has a list, and
                all lists have the same length.
            units (str): units of the system.
        """
        # if check_props:
        #     if do_copy:
        #         self.props = self._check_dictionary(deepcopy(props))
        #     else:
        #         self.props = self._check_dictionary(props)
        # else:
        #     self.props = props
        if props is not None and check_props:
            if do_copy:
                self.props = self._check_dictionary(deepcopy(props))
            else:
                self.props = self._check_dictionary(props)
        elif props is None:
            self.props = props

    def __len__(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return len(self.props["nxyz"])

    def __getitem__(self, idx):
        """Summary

        Args:
            idx (TYPE): Description

        Returns:
            TYPE: Description
        """
        return {key: val[idx] for key, val in self.props.items()}

    def __add__(self, other):
        """Summary

        Args:
            other (TYPE): Description

        Returns:
            TYPE: Description
        """
        if other.units != self.units:
            other = other.copy().to_units(self.units)

        new_props = self.props
        keys = list(new_props.keys())
        for key in keys:
            if key not in other.props:
                new_props.pop(key)
                continue
            val = other.props[key]
            if type(val) is list:
                new_props[key] += val
            else:
                old_val = new_props[key]
                new_props[key] = torch.cat([old_val, val.to(old_val.dtype)])
        self.props = new_props

        return copy.deepcopy(self)

    def _check_dictionary(self, props):
        """Check the dictionary or properties to see if it has the
        specified format.

        Args:
            props (TYPE): Description

        Returns:
            TYPE: Description
        """

        assert "nxyz" in props.keys()
        n_atoms = [len(x) for x in props["nxyz"]]
        n_geoms = len(props["nxyz"])

        if "num_atoms" not in props.keys():
            props["num_atoms"] = torch.LongTensor(n_atoms)
        else:
            props["num_atoms"] = torch.LongTensor(props["num_atoms"])

        for key, val in props.items():

            if val is None:
                props[key] = to_tensor([np.nan] * n_geoms)

            elif any([x is None for x in val]):
                bad_indices = [i for i, item in enumerate(val) if item is None]
                good_indices = [
                    index for index in range(len(val)) if index not in bad_indices
                ]
                if len(good_indices) == 0:
                    nan_list = np.array([float("NaN")]).tolist()
                else:
                    good_index = good_indices[0]
                    nan_list = (np.array(val[good_index]) * float("NaN")).tolist()
                for index in bad_indices:
                    props[key][index] = nan_list
                props.update({key: to_tensor(val)})

            else:
                assert len(val) == n_geoms, (
                    f"length of {key} is not "
                    f"compatible with {n_geoms} "
                    "geometries"
                )
                props[key] = to_tensor(val)

        return props

    def generate_neighbor_list(
        self, cutoff, undirected=True, key="nbr_list", offset_key="offsets"
    ):
        """Generates a neighbor list for each one of the atoms in the dataset.
            By default, does not consider periodic boundary conditions.

        Args:
            cutoff (float): distance up to which atoms are considered bonded.
            undirected (bool, optional): Description

        Returns:
            TYPE: Description
        """
        if "lattice" not in self.props:
            self.props[key] = [
                get_neighbor_list(nxyz[:, 1:4], cutoff, undirected)
                for nxyz in self.props["nxyz"]
            ]
            self.props[offset_key] = [
                torch.sparse.FloatTensor(nbrlist.shape[0], 3)
                for nbrlist in self.props[key]
            ]
        else:
            self._get_periodic_neighbor_list(
                cutoff=cutoff, undirected=undirected, offset_key=offset_key, nbr_key=key
            )
            return self.props[key], self.props[offset_key]

        return self.props[key]

    def make_all_directed(self):
        make_dset_directed(self)

    def generate_angle_list(self):

        if "lattice" in self.props:
            raise NotImplementedError("Angles not implemented for PBC.")

        self.make_all_directed()

        angles, nbrs = get_angle_list(self.props["nbr_list"])
        self.props["nbr_list"] = nbrs
        self.props["angle_list"] = angles

        ji_idx, kj_idx = add_ji_kj(angles, nbrs)

        self.props["ji_idx"] = ji_idx
        self.props["kj_idx"] = kj_idx

        return angles

    def generate_kj_ji(self, num_procs=1):
        """
        Generate only the `ji_idx` and `kj_idx` without storing
        the full angle list.
        """

        self.make_all_directed()
        add_kj_ji_parallel(self, num_procs=num_procs)

    def _get_periodic_neighbor_list(
        self, cutoff, undirected=False, offset_key="offsets", nbr_key="nbr_list"
    ):

        from nff.io.ase import AtomsBatch

        nbrlist = []
        offsets = []
        for nxyz, lattice in zip(self.props["nxyz"], self.props["lattice"]):
            atoms = AtomsBatch(
                nxyz[:, 0].long(),
                props={"num_atoms": torch.LongTensor([len(nxyz[:, 0])])},
                positions=nxyz[:, 1:],
                cell=lattice,
                pbc=True,
                cutoff=cutoff,
                directed=(not undirected),
            )
            nbrs, offs = atoms.update_nbr_list()
            nbrlist.append(nbrs)
            offsets.append(offs)

        self.props[nbr_key] = nbrlist
        self.props[offset_key] = offsets
        return

    def generate_bond_idx(self, num_procs=1):
        """
        For each index in the bond list, get the
        index in the neighbour list that corresponds to the
        same directed pair of atoms.
        Args:
            None
        Returns:
            None
        """

        self.make_all_directed()
        add_bond_idx_parallel(self, num_procs)

    def copy(self):
        """Copies the current dataset

        Returns:
            TYPE: Description
        """
        return Dataset(self.props, self.units)

    def change_idx(self, idx):
        """
        Change the dataset so that the properties are ordered by the
        indices `idx`. If `idx` does not contain all of the original
        indices in the dataset, then this will reduce the size of the
        dataset.
        """

        for key, val in self.props.items():
            if isinstance(val, list):
                self.props[key] = [val[i] for i in idx]
            else:
                self.props[key] = val[idx]

    def shuffle(self):
        """Summary

        Returns:
            TYPE: Description
        """
        idx = list(range(len(self)))
        reindex = skshuffle(idx)
        self.change_idx(reindex)

    def featurize(
        self,
        num_procs=NUM_PROCS,
        bond_feats=BOND_FEAT_TYPES,
        atom_feats=ATOM_FEAT_TYPES,
    ):
        """
        Featurize dataset with atom and bond features.
        Args:
            num_procs (int): number of parallel processes
            bond_feats (list[str]): names of bond features
            atom_feats (list[str]): names of atom features
        Returns:
            None
        """

        featurize_parallel(
            self, num_procs=num_procs, bond_feats=bond_feats, atom_feats=atom_feats
        )

    def add_morgan(self, vec_length):
        """
        Add Morgan fingerprints to each species in the dataset.
        Args:
            vec_length (int): length of fingerprint
        Returns:
            None
        """
        external_morgan(self, vec_length)

    def add_e3fp(self, fp_length, num_procs=NUM_PROCS):
        """
        Add E3FP fingerprints for each conformer of each species
        in the dataset.
        Args:
            fp_length (int): length of fingerprint
            num_procs (int): number of processes to use when
                featurizing.
        Returns:
            None
        """

        add_e3fp_parallel(self, fp_length, num_procs)

    def featurize_rdkit(self, method):
        """
        Add 3D-based RDKit fingerprints for each conformer of
        each species in the dataset.
        Args:
            method (str): name of RDKit feature method to use
        Returns:
            None
        """
        external_rdkit(self, method=method)

    def unwrap_xyz(self, mol_dic):
        """
        Unwrap molecular coordinates by displacing atoms by box vectors


        Args:
            mol_dic (dict): dictionary of nodes of each disconnected subgraphs
        """
        from nff.io.ase import AtomsBatch

        for i in range(len(self.props["nxyz"])):
            # makes atoms object

            atoms = AtomsBatch(
                positions=self.props["nxyz"][i][:, 1:4],
                numbers=self.props["nxyz"][i][:, 0],
                cell=self.props["cell"][i],
                pbc=True,
            )

            # recontruct coordinates based on subgraphs index
            if self.props["smiles"]:
                mol_idx = mol_dic[self.props["smiles"][i]]
                atoms.set_positions(reconstruct_atoms(atoms, mol_idx))
                nxyz = atoms.get_nxyz()
            self.props["nxyz"][i] = torch.Tensor(nxyz)

    def save(self, path):
        """Summary

        Args:
            path (TYPE): Description
        """

        # to deal with the fact that sparse tensors can't be pickled
        offsets = self.props.get("offsets", torch.LongTensor([0]))
        old_offsets = copy.deepcopy(offsets)

        # check if it's a sparse tensor. The first two conditions
        # Are needed for backwards compatability in case it's a float
        # or empty list

        if all([hasattr(offsets, "__len__"), len(offsets) > 0]):
            if isinstance(offsets[0], torch.sparse.FloatTensor):
                self.props["offsets"] = [val.to_dense() for val in offsets]

        torch.save(self, path)
        if "offsets" in self.props:
            self.props["offsets"] = old_offsets

    def gen_bond_stats(self):

        bond_len_dict = {}
        # generate bond statistics
        for i in range(len(self.props["nxyz"])):
            z = self.props["nxyz"][i][:, 0]
            xyz = self.props["nxyz"][i][:, 1:4]
            bond_list = self.props["bonds"][i]
            bond_len = (
                (xyz[bond_list[:, 0]] - xyz[bond_list[:, 1]])
                .pow(2)
                .sum(-1)
                .sqrt()[:, None]
            )

            bond_type_list = torch.stack((z[bond_list[:, 0]], z[bond_list[:, 1]])).t()
            for i, bond in enumerate(bond_type_list):
                bond = tuple(torch.LongTensor(sorted(bond)).tolist())
                if bond not in bond_len_dict.keys():
                    bond_len_dict[bond] = [bond_len[i]]
                else:
                    bond_len_dict[bond].append(bond_len[i])

        # compute bond len averages
        self.bond_len_dict = {
            key: torch.stack(bond_len_dict[key]).mean(0) for key in bond_len_dict.keys()
        }

        return self.bond_len_dict

    def gen_bond_prior(self, cutoff, bond_len_dict=None):
        from nff.io.ase import AtomsBatch

        if not self.props:
            raise TypeError("the dataset has no data yet")

        bond_dict = {}
        mol_idx_dict = {}

        # ---------This part can be simplified---------#
        for i in range(len(self.props["nxyz"])):
            z = self.props["nxyz"][i][:, 0]
            xyz = self.props["nxyz"][i][:, 1:4]

            # generate arguments for ase Atoms object
            cell = self.props["cell"][i] if "cell" in self.props.keys() else None
            ase_param = {"numbers": z, "positions": xyz, "pbc": True, "cell": cell}

            atoms = Atoms(**ase_param)
            sys_name = self.props["smiles"][i]
            if sys_name not in bond_dict.keys():
                print(sys_name)
                i, j = neighbor_list("ij", atoms, DISTANCETHRESHOLDICT_Z)

                bond_list = torch.LongTensor(np.stack((i, j), axis=1)).tolist()
                bond_dict[sys_name] = bond_list

                # generate molecular graph
                # TODO: there is redundant code in generate_subgraphs
                subgraph_index = generate_subgraphs(atoms)
                mol_idx_dict[sys_name] = subgraph_index

        # generate topologies
        # TODO: include options to only generate bond topology
        self.generate_topologies(bond_dic=bond_dict)
        if "cell" in self.props.keys():
            self.unwrap_xyz(mol_idx_dict)
        # ---------This part can be simplified---------#

        # generate bond length dictionary if not given
        if not bond_len_dict:
            bond_len_dict = self.gen_bond_stats()

        # update bond len and offsets
        all_bond_len = []
        all_offsets = []
        all_nbr_list = []
        for i in range(len(self.props["nxyz"])):
            z = self.props["nxyz"][i][:, 0]
            xyz = self.props["nxyz"][i][:, 1:4]

            bond_list = self.props["bonds"][i]
            bond_type_list = torch.stack((z[bond_list[:, 0]], z[bond_list[:, 1]])).t()
            bond_len_list = []
            for bond in bond_type_list:
                bond_type = tuple(torch.LongTensor(sorted(bond)).tolist())
                bond_len_list.append(bond_len_dict[bond_type])
            all_bond_len.append(torch.Tensor(bond_len_list).reshape(-1, 1))

            # update offsets
            cell = (self.props["cell"][i] if "cell" in self.props.keys() else None,)
            ase_param = {
                "numbers": z,
                "positions": xyz,
                "pbc": True,
                "cutoff": cutoff,
                "cell": cell,
                "nbr_torch": False,
            }

            # the coordinates have been unwrapped and try to results offsets
            atoms = AtomsBatch(**ase_param)
            atoms.update_nbr_list()
            all_offsets.append(atoms.offsets)
            all_nbr_list.append(atoms.nbr_list)

        # update
        self.props["bond_len"] = all_bond_len
        self.props["offsets"] = all_offsets
        self.props["nbr_list"] = all_nbr_list
        self._check_dictionary(deepcopy(self.props))

    @classmethod
    def load(cls, path):
        """Summary

        Args:
            path (TYPE): Description

        Returns:
            TYPE: Description

        Raises:
            TypeError: Description
        """
        obj = torch.load(path)
        if isinstance(obj, cls):
            return obj
        else:
            raise TypeError("{} is not an instance from {}".format(path, type(cls)))


def convert_nan(x):
    """
    If a list has any elements that contain nan, convert its contents
    to the right form so that it can eventually be converted to a tensor.
    Args:
        x (list): any list with floats, ints, or Tensors.
    Returns:
        new_x (list): updated version of `x`
    """

    new_x = []
    # whether any of the contents have nan
    has_nan = any([np.isnan(y).any() for y in x])
    for y in x:

        if has_nan:
            # if one is nan then they will have to become float tensors
            if type(y) in [int, float]:
                new_x.append(torch.Tensor([y]))
            elif isinstance(y, torch.Tensor):
                new_x.append(y.float())
            elif isinstance(y, list):
                new_x.append(torch.Tensor(y))
            else:
                msg = (
                    "Don't know how to convert sub-components of type "
                    f"{type(x)} when components might contain nan"
                )
                raise Exception(msg)
        else:
            # otherwise they can be kept as is
            new_x.append(y)

    return new_x


def to_tensor(x, stack=False):
    """
    Converts input `x` to torch.Tensor.
    Args:
        x (list of lists): input to be converted. Can be: number, string, list, array, tensor
        stack (bool): if True, concatenates torch.Tensors in the batching dimension
    Returns:
        torch.Tensor or list, depending on the type of x
    Raises:
        TypeError: Description
    """

    # a single number should be a list
    if isinstance(x, numbers.Number):
        return torch.Tensor([x])

    if isinstance(x, str):
        return [x]

    if isinstance(x, torch.Tensor):
        return x

    if type(x) is list and type(x[0]) != str:
        if not isinstance(x[0], torch.sparse.FloatTensor):
            x = convert_nan(x)

    # all objects in x are tensors
    if isinstance(x, list) and all([isinstance(y, torch.Tensor) for y in x]):

        # list of tensors with zero or one effective dimension
        # flatten the tensor

        if all([len(y.shape) < 1 for y in x]):
            return torch.cat([y.view(-1) for y in x], dim=0)

        elif stack:
            return torch.cat(x, dim=0)

        # list of multidimensional tensors
        else:
            return x

    # some objects are not tensors
    elif isinstance(x, list):

        # list of strings
        if all([isinstance(y, str) for y in x]):
            return x

        # list of ints
        if all([isinstance(y, int) for y in x]):
            return torch.LongTensor(x)

        # list of floats
        if all([isinstance(y, numbers.Number) for y in x]):
            return torch.Tensor(x)

        # list of arrays or other formats
        if any([isinstance(y, (list, np.ndarray)) for y in x]):
            return [torch.Tensor(y) for y in x]

    raise TypeError("Data type not understood")
