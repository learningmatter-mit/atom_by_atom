from pymatgen.electronic_structure.core import OrbitalType
from pymatgen.core.periodic_table import Element
import numpy as np

# utilities for working with ProjectedDOS objects

def is_conducting(dos, tol=0.001):

    # checks if a structure is conducting by evaluating if the fermi level is below the valence band maximum (within a tolerance tol)
    if dos.efermi-tol > dos.vbm:
        return False

    return True

def get_site_valence_orbital_dos(dos, site):

    # takes a projectedDOS object and a site
    # returns the dos projected onto both the site and the valence orbital for that site element

    el = Element(list(dos.structure.sites[site].species.as_dict().keys())[0])
    orbital_type = OrbitalType(np.min([2, el.valence[0]])) # orbital type matches valence angular momentum (but cannot go higher than d)

    cdos = dos.as_pymatgen_pdos()
    return cdos.get_site_spd_dos(dos.structure.sites[site])[orbital_type]

def get_n_moment(x, y, n):

    return np.trapz(x**n * y, x=x) / np.trapz(y, x=x)

def get_site_valence_orbital_band_value(dos, site, value):

    # given a projectedDOS object and a site, return the value of the band
    # options for value: center, width (skew+kertusos not yet implemented)
    
    pdos = get_site_valence_orbital_dos(dos, site) 
    energies = pdos.energies
    densities = sum(list(pdos.densities.values()))
    
    index_min = np.argmax(energies > pdos.efermi-5)
    index_max = np.argmax(energies > pdos.efermi+2)

    energies = energies[index_min:index_max]
    densities = densities[index_min:index_max]
   
    if value=='center':
        return get_n_moment(energies, densities, n=1)
    elif value=='width':
        return np.sqrt(get_n_moment(energies, densities, n=2))
    else:
        raise NotImplementedError
