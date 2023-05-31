from pymatgen.electronic_structure.core import OrbitalType
from pymatgen.core.periodic_table import Element
import numpy as np

from scipy import integrate

# utilities for working with ProjectedDOS objects

erange = [-10,2]

def is_conducting(dos, tol=0.001):

    if dos == None:
        return 'unknown'

    # checks if a structure is conducting by evaluating if the fermi level is below the valence band maximum (within a tolerance tol)
    if dos.efermi-tol > dos.vbm:
        return False

    return True

def orbital_type(el):

    # return the valence (s=0, p=1, d=2) 
    # NOTE: returns d when valence orbital type is f (f-band centers not available)
    electronic_structure = el.full_electronic_structure
    orbital_types = np.array(electronic_structure)[:,1]

    if 'd' in orbital_types:
        return 2
    elif 'p' in orbital_types:
        return 1
    else:
        return 0

def neutral_valence(el):

    # return the number of electrons in the valence shell when element is neutral
    # NOTE: returns number of electrons in d when valence orbital is f (f-band centers not available)
    electronic_structure = el.full_electronic_structure
    orbital_types = np.array(electronic_structure)[:,1]

    if 'd' in orbital_types:
        orb_info = electronic_structure[np.where(['d' in orbital for orbital in orbital_types])[0][-1]]
    elif 'p' in orbital_types:
        orb_info = electronic_structure[np.where(['p' in orbital for orbital in orbital_types])[0][-1]]
    else:
        orb_info = electronic_structure[np.where(['s' in orbital for orbital in orbital_types])[0][-1]]

    return orb_info[2]

def get_site_valence_dos(dos, site):

    # takes a site and a ProjectedDOS object and returns the dos projected onto the site and valence orbitals
    el = Element(list(dos.structure.sites[site].species.as_dict().keys())[0])
    otype = OrbitalType(orbital_type(el))

    cdos = dos.as_pymatgen_pdos()
    pdos = cdos.get_site_spd_dos(dos.structure.sites[site])[otype]

    return pdos.energies-pdos.efermi, sum(list(pdos.densities.values())) # returnes energies (shifted w.r.t. fermi level), densities

def get_site_valence_bandcenter(dos, site):

    # gets the bandcenter with respect to the fermi level for a given site (projected onto valence orbitals)
    # using the same limits as in the surface science paper (-10, 2) with respect to the fermi level

    el = Element(list(dos.structure.sites[site].species.as_dict().keys())[0])
    otype = OrbitalType(orbital_type(el)) 

    cdos = dos.as_pymatgen_pdos()
    return cdos.get_band_center(band=otype, sites=[dos.structure.sites[site]], erange=erange) 


def get_site_valence_bandwidth(dos, site):
    
    # gets the bandwidth with respect to the fermi level for a given site (projected onto valence orbitals)
    # using the same limits as in the surface science paper (-10, 2) with respect to the fermi level

    el = Element(list(dos.structure.sites[site].species.as_dict().keys())[0])
    otype = OrbitalType(orbital_type(el)) # orbital type matches valence angular momentum (but cannot go higher than d)
    
    cdos = dos.as_pymatgen_pdos()
    return cdos.get_band_width(band=otype, sites=[dos.structure.sites[site]], erange=erange) 
