#!/usr/bin/env python
# coding: utf-8

import sys
import os
import django
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import shutil
from matplotlib import colors
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import seaborn as sns
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from pymatgen.core.periodic_table import *

sys.path.append('/home/lungerja/projects/htvs/')
sys.path.append('/home/lungerja/projects/htvs/djangochem/')

# setup the djhango settings file.  Change this to use the settings file that connects you to your desired database
os.environ["DJANGO_SETTINGS_MODULE"] = "djangochem.settings.orgel"

# this must be run to setup access to the django settings and make database access work etc.
django.setup()

#from imports import *
#from django_imports import *
from pymatgen.core.periodic_table import Element
from chemconfigs.vasp.defaults import Magmom

from ase.visualize import view
from catkit.gen.utils import *
from ase import Atoms

from jobs.models import Job, JobConfig
from django.contrib.contenttypes.models import ContentType
from ase.io import write
from pgmols.models import *
from docking.models import *

from analysis.structure.perovskite import PerovskiteAnalyzer
from pymatgen.core.periodic_table import Element

from pymatgen.electronic_structure.core import OrbitalType
from scipy import integrate

group = Group.objects.get(name='perovskite')
opt_config = JobConfig.objects.get(name='pbe_u_paw_spinpol_opt_vasp')
es_config = JobConfig.objects.get(name='pbe_u_paw_spinpol_es_vasp')
morgan_parse_config = JobConfig.objects.get(name='morgan_gga_u_parse')

class CrystalAnalyzer():
    '''
    Analyzer class that takes in a crystal object and returns values of interest
    it is assumed that the crystals being analyzed with this class have the job config pbe_u_paw_spinpol_opt_vasp
    '''
    def __init__(
            self,
            crystal,
    ):

        # general items of interest
        self.crystal = crystal
        self.id_ = crystal.id

        # site properties of interest
        self.magmoms = self.get_site_magmoms()
        self.site_bandcenters = self.get_site_bandcenters()
    
    def get_projected_dos(self):
        '''
        Try to get associated ProjectedDOS object, and if not available returns None 
        '''
        es_parent_job = self.crystal.childjobs.get(config__name__contains = "pbe_u_paw_spinpol_es", status='done')
        pdos = ProjectedDOS.objects.get(parentjob = es_parent_job)
        return pdos

    def get_band_center(self, pdos):
        '''
        pdos: pymatgen DOS object
    
        Returns:
        band_center of the dos
        '''
        energies = pdos.energies
        densities = sum(list(pdos.densities.values()))
        normalization = integrate.simps(densities, energies)
        band_center = integrate.simps(densities*energies, energies)
    
        return band_center / normalization - pdos.efermi

    def get_site_bandcenters(self):
        '''
        if ProjectedDOS exists for this surface, this method will return the site bandcenters. Otherwise returns None
        '''
        try: 
            pdos = self.get_projected_dos()
        except:
            return None

        complete_dos = pdos.as_pymatgen_pdos()
        orbital_types = [OrbitalType(0), OrbitalType(1), OrbitalType(2)]
        orbital_site_bandcenters = {key:[] for key in orbital_types}
        for site in complete_dos.structure:
            site_spd_dos = complete_dos.get_site_spd_dos(site)
            for orbital_type, orbital_list in orbital_site_bandcenters.items():
                dos = site_spd_dos[orbital_type]
                bandcenter = self.get_band_center(dos)
                orbital_list.append(bandcenter)

        site_bandcenters = {key.name:value for key, value in orbital_site_bandcenters.items()}
    
        return site_bandcenters


    def get_site_magmoms(self):

        magmoms = self.crystal.calcs.filter(props__magmoms__isnull=False)
        if magmoms.exists():
            return magmoms.values_list('props__magmoms',flat=True)[0]
        else:
            return None


    def get_oxygen_sites(self):

        structure = self.crystal.as_pymatgen_structure() 
        symbols = [spec.symbol for spec in structure.species]
        return np.where([symbol=='O' for symbol in symbols])[0]
    
    def get_metal_sites(self):

        structure = self.crystal.as_pymatgen_structure() 
        symbols = [spec.symbol for spec in structure.species]
        return np.where([symbol!='O' for symbol in symbols])[0]

    def oxygen_O2p(self):

        return np.array(self.site_bandcenters['p'])[self.get_oxygen_sites()]
    
    def metal_d(self):

        return np.array(self.site_bandcenters['d'])[self.get_metal_sites()]


class SurfaceAnalyzer(CrystalAnalyzer):
    '''
    Analyzer class that takes in a surface object and returns values of interest
    it is assumed that the surfaces being analyzed with this class have the job config pbe_u_paw_spinpol_opt_surf_vasp
    '''
    def __init__(
            self,
            crystal,
    ):

        super(SurfaceAnalyzer, self).__init__(crystal)

        self.adsorbate = self.get_adsorbate()

        # indices of sites of interest
        #self.surface_metal_sites = self.get_surface_metal_sites()
        #self.surface_oxygen_sites = self.get_surface_oxygen_sites()
        #self.active_sites = self.get_active_sites()

    def get_adsorbate(self):

        # get the stoichiometry of the surface adsorbate
        adsorbate_atoms = self.crystal.adsorbate_atoms

        atomic_numbers = np.array(self.crystal.xyz)[:,0].tolist()
        atomic_symbols = [Element.from_Z(z).symbol for z in atomic_numbers]

        adsorbate_symbols = [atomic_symbols[i] for i in np.where(adsorbate_atoms)[0]]
        adsorbate_symbols.sort()
        return "".join(adsorbate_symbols)

