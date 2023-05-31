# Utilities for working with Surface objects

from pymatgen.core.periodic_table import Element
import numpy as np

def remove_values_from_list(the_list, val):
    return [value for value in the_list if value != val]

def get_adsorbate_sites(surface):

    adsorbate_atoms = surface.adsorbate_atoms
    return np.where(adsorbate_atoms)[0]

def get_adsorbate(surface):

    atomic_numbers = np.array(surface.xyz)[:,0].tolist()
    atomic_symbols = [Element.from_Z(z).symbol for z in atomic_numbers]

    adsorbate_symbols = [atomic_symbols[i] for i in get_adsorbate_sites(surface)]
    adsorbate_symbols.sort()
    return "".join(adsorbate_symbols)

def get_surface_sites(surface):
    
    surface_atoms = surface.surface_atoms
    return np.where(surface_atoms)[0]

def get_O_adsorbate_sites(surface):
    
    adsorbate_sites = get_adsorbate_sites(surface)
    atomic_numbers = np.array(surface.xyz)[:,0].tolist()
    atomic_symbols = [Element.from_Z(z).symbol for z in atomic_numbers]

    O_adsorbate_sites=[]
    for adsorbate_site in adsorbate_sites:
        if np.array(atomic_symbols)[adsorbate_site] == 'O':
            O_adsorbate_sites.append(adsorbate_site)

    return O_adsorbate_sites

def get_O_surface_sites(surface):
    
    surface_sites = get_surface_sites(surface)
    atomic_numbers = np.array(surface.xyz)[:,0].tolist()
    atomic_symbols = [Element.from_Z(z).symbol for z in atomic_numbers]

    O_surface_sites=[]
    for surface_site in surface_sites:
        if np.array(atomic_symbols)[surface_site] == 'O':
            O_surface_sites.append(surface_site)

    return O_surface_sites

def is_top_site(surface):

    O_adsorbate_site = get_O_adsorbate_sites(surface)[0]

    atomic_numbers = np.array(surface.xyz)[:,0].tolist()
    atomic_symbols = np.array([Element.from_Z(z).symbol for z in atomic_numbers])

    structure = surface.as_pymatgen_structure()
    neighbors = structure.get_neighbors(structure.sites[O_adsorbate_site],2.5)
    neighbors = list(set([n.index for n in neighbors]))

    neighbor_symbols = atomic_symbols[neighbors]
    neighbor_symbols = remove_values_from_list(neighbor_symbols, 'H')
    neighbor_symbols = remove_values_from_list(neighbor_symbols, 'O')

    # check that the number of neighbors is consistent with this being a top site
    if len(neighbor_symbols) == 1:
        return True

    return False

def get_active_site(surface):

    # CAUTION!! CURRENTLY ONLY IMPLEMENTED FOR O AND OH
    # check neighbors of adsorbed oxygen and make sure it is only H and the active site
    O_adsorbate_site = get_O_adsorbate_sites(surface)[0]
    
    atomic_numbers = np.array(surface.xyz)[:,0].tolist()
    atomic_symbols = [Element.from_Z(z).symbol for z in atomic_numbers]

    structure = surface.as_pymatgen_structure()
    neighbors = structure.get_neighbors(structure.sites[O_adsorbate_site],2.5)
    
    for neighbor in neighbors:
        if atomic_symbols[neighbor.index]!='O' and atomic_symbols[neighbor.index]!='H':
            return neighbor.index

    return None

def get_active_species(surface):

    # CAUTION!! CURRENTLY ONLY IMPLEMENTED FOR O AND OH
    # check neighbors of adsorbed oxygen and make sure it is only H and the active site
    active_site = get_active_site(surface)

    if active_site is None:
        return None

    atomic_numbers = np.array(surface.xyz)[:,0].tolist()
    atomic_symbols = [Element.from_Z(z).symbol for z in atomic_numbers]

    return Element(atomic_symbols[active_site])
