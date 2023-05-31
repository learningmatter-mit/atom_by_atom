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
from pymatgen.core.structure import *

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

# dos utilities
def get_Ev(dos, el, bader):

    # get energy valence, the energy at which all the valence electrons for this element/per-site dos are accounted for
    energies = dos.energies
    densities = sum(list(dos.densities.values()))

    # get index of fermi energy
    index_fermi = np.argmax(energies > dos.efermi)


def get_band_center(dos, Ev):
    '''
    pdos: pymatgen DOS object

    Returns:
    band_center of the dos
    '''
    energies = dos.energies
    densities = sum(list(dos.densities.values()))

    # only include energies between -10 and 2 eV relative to the fermi level (like Surface Science paper)
    #index_min = np.argmax(energies > dos.efermi-10)
    #index_max = np.argmax(energies > dos.efermi+2)

    # integrate down from the fermi level until all electrons participating in reaction are accounted for (neutral number of valence electrons - bader charge)



    energies = energies[index_min:index_max]
    densities = densities[index_min:index_max]
    
    normalization = integrate.simps(densities, energies)
    band_center = integrate.simps(densities*energies, energies)

    return band_center / normalization - dos.efermi


class SurfacePDOSAnalyzer():
    '''
    Analyzer class that takes in a surface object and returns values of interest
    it is assumed that the surfaces being analyzed with this class have the job config pbe_u_paw_spinpol_opt_surf_vasp
    '''
    def __init__(
            self,
            id_,  
    ):

        self.opt_config = 'pbe_u_paw_spinpol_opt_surf_vasp'

        self.id_ = id_
        self.pdos = ProjectedDOS.objects.filter(parentjob__id = id_).last()
        self.structure = self.pdos.structure
        
        self.surface = self.pdos.parentjob.parent
        self.adsorbate = self.get_adsorbate()

        try:
            self.B = self.surface.bulk.details['B'][0]  # for now just take the first one, need to re-do this for multi b site
        except:
            self.B = None
    
    def get_parent(self, config_name):

        parent = self.pdos.parentjob.parent
        while parent.parentjob.config.name!=config_name:
            parent = parent.parentjob.parent
        return parent

    def is_conducting(self, tol=0.001):

        # checks if a structure is conducting by evaluating if the fermi level is below the valence band maximum (within a tolerance tol)
        if self.pdos.efermi-tol > self.pdos.vbm:
            return False

        return True

    def get_site_bandcenters(self):
        '''
        if ProjectedDOS exists for this surface, this method will return the site bandcenters. Otherwise returns None
        '''
        complete_dos = self.pdos.as_pymatgen_pdos()
        orbital_types = [OrbitalType(0), OrbitalType(1), OrbitalType(2)]
        orbital_site_bandcenters = {key:[] for key in orbital_types}
        for site in complete_dos.structure:
            site_spd_dos = complete_dos.get_site_spd_dos(site)
            for orbital_type, orbital_list in orbital_site_bandcenters.items():
                dos = site_spd_dos[orbital_type]
                bandcenter = get_band_center(dos)
                orbital_list.append(bandcenter)

        site_bandcenters = {key.name:value for key, value in orbital_site_bandcenters.items()}
    
        return site_bandcenters

    def get_site_bader_charges(self):

        # get optimized parent
        opt_parent = self.get_parent(self.opt_config)
        bader_charges = opt_parent.calcs.filter(props__atomiccharges__isnull=False)
        if bader_charges.exists():
            return bader_charges.values_list('props__atomiccharges',flat=True)[0]['bader']
        return None

    def structure_with_site_properties(self):

        site_properties = self.get_site_bandcenters()
        site_properties.update({'bader':self.get_site_bader_charges()})

        return self.structure.copy(site_properties=site_properties)

    def get_adsorbate(self):
        
        adsorbate_atoms = self.surface.adsorbate_atoms

        atomic_numbers = np.array(self.surface.xyz)[:,0].tolist()
        atomic_symbols = [Element.from_Z(z).symbol for z in atomic_numbers]

        adsorbate_symbols = [atomic_symbols[i] for i in np.where(adsorbate_atoms)[0]]
        adsorbate_symbols.sort()
        return "".join(adsorbate_symbols)

    def get_adsorbate_sites(self):

        adsorbate_atoms = self.surface.adsorbate_atoms
        return np.where(adsorbate_atoms)[0]

    def get_surface_sites(self):
        
        surface_atoms = self.surface.surface_atoms
        return np.where(surface_atoms)[0]

    def get_O_adsorbate_sites(self):
        
        adsorbate_sites = self.get_adsorbate_sites()
        atomic_numbers = np.array(self.surface.xyz)[:,0].tolist()
        atomic_symbols = [Element.from_Z(z).symbol for z in atomic_numbers]

        O_adsorbate_sites=[]
        for adsorbate_site in adsorbate_sites:
            if np.array(atomic_symbols)[adsorbate_site] == 'O':
                O_adsorbate_sites.append(adsorbate_site)

        return O_adsorbate_sites
    
    def get_O_surface_sites(self):
        
        surface_sites = self.get_surface_sites()
        atomic_numbers = np.array(self.surface.xyz)[:,0].tolist()
        atomic_symbols = [Element.from_Z(z).symbol for z in atomic_numbers]

        O_surface_sites=[]
        for surface_site in surface_sites:
            if np.array(atomic_symbols)[surface_site] == 'O':
                O_surface_sites.append(surface_site)

        return O_surface_sites

    def is_top_site(self):

        # CAUTION!! CURRENTLY ONLY IMPLEMENTED FOR O AND OH
        # check neighbors of adsorbed oxygen and make sure it is only H and the active site
        O_adsorbate_site = self.get_O_adsorbate_sites()[0]
        
        atomic_numbers = np.array(self.surface.xyz)[:,0].tolist()
        atomic_symbols = [Element.from_Z(z).symbol for z in atomic_numbers]

        structure = self.surface.as_pymatgen_structure()
        neighbors = structure.get_neighbors(structure.sites[O_adsorbate_site],2.5)
        neighbor_symbols = [atomic_symbols[neighbor.index] for neighbor in neighbors]
        
        # check the adsorbed O isn't attached to something weird like an oxygen
        for symbol in neighbor_symbols:
            if symbol!='H' and symbol!=self.B:
                return False
       
        # check that the number of neighbors is consistent with this being a top site
        if len(neighbor_symbols) != len(self.adsorbate):
            return False

        return True
    
    
    def get_active_site(self):

        # CAUTION!! CURRENTLY ONLY IMPLEMENTED FOR O AND OH
        # check neighbors of adsorbed oxygen and make sure it is only H and the active site
        O_adsorbate_site = self.get_O_adsorbate_sites()[0]
        
        atomic_numbers = np.array(self.surface.xyz)[:,0].tolist()
        atomic_symbols = [Element.from_Z(z).symbol for z in atomic_numbers]

        structure = self.surface.as_pymatgen_structure()
        neighbors = structure.get_neighbors(structure.sites[O_adsorbate_site],2.5)
        
        for neighbor in neighbors:
            if atomic_symbols[neighbor.index]==self.B:
                return neighbor.index
    
        return None

# plotting utilities

def plot_hexbin(targ, pred, key, title="", scale="linear", 
                inc_factor = 1.1, dec_factor = 0.9,
                bins=None, plot_helper_lines=False,
                cmap='viridis', gridsize=100):
    
    props = {
        'center_diff': 'B 3d $-$ O 2p difference',
        'op': 'O 2p $-$ $E_v$',
        'form_e': 'formation energy',
        'e_hull': 'energy above hull',
        'tot_e': 'energy per atom',
        'time': 'runtime',
        'magmom': 'magnetic moment',
        'magmom_abs': 'magnetic moment',
        'ads_e': 'adsorption energy',
        'acid_stab': 'electrochemical stability',
        'bandcenter': 'DOS band center',
        'phonon': 'atomic vibration frequency',
        'bader': 'Bader charge'
    }
    
    units = {
        'center_diff': 'eV',
        'op': 'eV',
        'form_e': 'eV',
        'e_hull': 'eV/atom',
        'tot_e': 'eV/atom',
        'time': 's',
        'magmom': '$\mu_B$',
        'magmom_abs': '|$\mu_B$|',
        'ads_e': 'eV',
        'acid_stab': 'eV/atom',
        'bandcenter': 'eV',
        'phonon': 'THz',
        'bader': '$q_e$'
    }
    
    fig, ax = plt.subplots(figsize=(6,6))
    
    mae = mean_absolute_error(targ, pred)
    r, _ = pearsonr(targ, pred)
    
    if scale == 'log':
        pred = np.abs(pred) + 1e-8
        targ = np.abs(targ) + 1e-8
        
    lim_min = min(np.min(pred), np.min(targ))
    if lim_min < 0:
        if lim_min > -0.1:
            lim_min = -0.1
        lim_min *= inc_factor
    else:
        if lim_min < 0.1:
            lim_min = -0.1
        lim_min *= dec_factor
    lim_max = max(np.max(pred), np.max(targ))
    if lim_max <= 0:
        if lim_max > -0.1:
            lim_max = 0.2
        lim_max *= dec_factor
    else:
        if lim_max < 0.1:
            lim_max = 0.25
        lim_max *= inc_factor
        
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect('equal')

    #ax.plot((lim_min, lim_max),
    #        (lim_min, lim_max),
    #        color='#000000',
    #        zorder=-1,
    #        linewidth=0.5)
    ax.axline((0, 0), (1, 1),
           color='#000000',
           zorder=-1,
           linewidth=0.5)
       
    hb = ax.hexbin(
        targ, pred,
        cmap=cmap,
        gridsize=gridsize,
        bins=bins,
        mincnt=1,
        edgecolors=None,
        linewidths=(0.1,),
        xscale=scale,
        yscale=scale,
        extent=(lim_min, lim_max, lim_min, lim_max))
    

    cb = fig.colorbar(hb, shrink=0.822)
    cb.set_label('Count')

    if plot_helper_lines:
        
        if scale == 'linear':
            x = np.linspace(lim_min, lim_max, 50)
            y_up = x + mae
            y_down = x - mae         
            
        elif scale == 'log':
            x = np.logspace(np.log10(lim_min), np.log10(lim_max), 50)
            
            # one order of magnitude
            y_up = np.maximum(x + 1e-2, x * 10)
            y_down = np.minimum(np.maximum(1e-8, x - 1e-2), x / 10)
            
            # one kcal/mol/Angs
            y_up = x + 1
            y_down = np.maximum(1e-8, x - 1)
            
        
        for y in [y_up, y_down]:
            ax.plot(x,
                    y,
                    color='#000000',
                    zorder=2,
                    linewidth=0.5,
                    linestyle='--')
            
    ax.set_title(title, fontsize=14)
    ax.set_ylabel('Predicted %s [%s]' % (props[key], units[key]), fontsize=12)
    ax.set_xlabel('Calculated %s [%s]' % (props[key], units[key]), fontsize=12)
    
    ax.annotate("Pearson's r: %.3f \nMAE: %.3f %s " % (r, mae, units[key]),
                (0.03, 0.88),
                xycoords='axes fraction',
                fontsize=12)
         
    return r, mae, ax, hb

def flatten(t):
    return [item for sublist in t for item in sublist]

def violin_plot(test_ids, structures, test_targs, test_preds, color1, color2, ax, Z_range=[21,31]):

    pred_dictionary={}
    dictionary={}

    #plt.figure(figsize=(5,5))
    
    for index in tqdm(range(len(test_ids))):
       
        id_ = test_ids[index]
        structure = structures[id_]
        elems = [Element(x.symbol).Z for x in structure.species]

        pred = test_preds[index]#.numpy()
        targ = test_targs[index]#.numpy()

        # get dictionary
        for i in range(len(elems)):

            elem = elems[i]
            y = targ[i]

            if elem in dictionary:
                array = dictionary[elem]
                array.append(y)
                dictionary[elem] = array

            else:
                dictionary[elem] = [y]
        
        # get pred_dictionary
        for i in range(len(elems)):

            elem = elems[i]
            y = pred[i]

            if elem in pred_dictionary:
                array = pred_dictionary[elem]
                array.append(y)
                pred_dictionary[elem] = array

            else:
                pred_dictionary[elem] = [y]

    df = pd.DataFrame(columns=('Z', 'y','hue'))
    Zs = []
    ys = []
    hues = []

    for Z in range(Z_range[0], Z_range[1]):

        ys.append(dictionary[Z])
        Zs.append([Z for i in range(len(dictionary[Z]))])
        hues.append(['targ' for i in range(len(dictionary[Z]))])

        ys.append(pred_dictionary[Z])
        Zs.append([Z for i in range(len(pred_dictionary[Z]))])
        hues.append(['pred' for i in range(len(pred_dictionary[Z]))])

    Zs = flatten(Zs)
    ys = flatten(ys)
    hues = flatten(hues)

    df['Z'] = Zs
    df['y'] = ys
    df['hue'] = hues
    
    my_pal = {"targ": color1, "pred": color2}

    ax = sns.violinplot(x="Z", y="y", hue="hue",
                    data=df, palette=my_pal, split=True, inner=None,linewidth=1)
   
    #plt.ylim([-0.1,5.2])
    plt.legend(loc='upper right')

    xticks = [Element.from_Z(int(text._text)).symbol for text in ax.get_xticklabels()]
    ax.set_xticklabels(xticks)

    return ax
