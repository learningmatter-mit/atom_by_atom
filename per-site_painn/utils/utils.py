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

class BulkPDOSAnalyzer():
    '''
    Analyzer class that takes in an es job and returns values of interest
    it is assumed that the crystals being analyzed with this class have the job config pbe_u_paw_spinpol_opt_vasp
    '''
    def __init__(
            self,
            id_,  
    ):

        self.id_ = id_
        self.pdos = ProjectedDOS.objects.filter(parentjob__id = id_).last()
        self.structure = self.pdos.structure
        self.site_bandcenters = self.get_site_bandcenters()

    
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
        complete_dos = self.pdos.as_pymatgen_pdos()
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

    def structure_with_bandcenters(self):

        site_bandcenters = self.get_site_bandcenters()
        return self.structure.copy(site_properties=site_bandcenters)

    def get_oxygen_sites(self):

        symbols = [spec.symbol for spec in self.structure.species]
        return np.where([symbol=='O' for symbol in symbols])[0]

    def get_metal_sites(self):
    
        symbols = [spec.symbol for spec in self.structure.species]
        return np.where([symbol!='O' for symbol in symbols])[0]
    
    def get_active_sites(self):

        Bs = Job.objects.get(id=self.id_).parent.details['B']
        symbols = [spec.symbol for spec in self.structure.species]
        return np.where([symbol in Bs for symbol in symbols])[0]

    def oxygen_O2p(self):

        return np.array(self.site_bandcenters['p'])[self.get_oxygen_sites()]
    
    def metal_d(self):

        return np.array(self.site_bandcenters['d'])[self.get_metal_sites()]
    
    def active_metal_d(self):

        return np.array(self.site_bandcenters['d'])[self.get_active_sites()]


class SurfacePDOSAnalyzer(BulkPDOSAnalyzer):
    '''
    Analyzer class that takes in a surface object and returns values of interest
    it is assumed that the surfaces being analyzed with this class have the job config pbe_u_paw_spinpol_opt_surf_vasp
    '''
    def __init__(
            self,
            id_,
    ):

        super(SurfacePDOSAnalyzer, self).__init__(id_)

        self.surface = self.pdos.parentjob.parent
        try:
            self.B = self.surface.bulk.details['B'][0]  # for now just take the first one, need to re-do this for multi b site
        except:
            self.B = None

    def get_parent(self, config_name):

        parent = self.surface
        while parent.parentjob.config.name!=config_name:
            parent = parent.parentjob.parent

        return parent

    def get_adsorbate(self):
        
        adsorbate_atoms = self.surface.adsorbate_atoms

        atomic_numbers = np.array(self.surface.xyz)[:,0].tolist()
        atomic_symbols = [Element.from_Z(z).symbol for z in atomic_numbers]

        adsorbate_symbols = [atomic_symbols[i] for i in np.where(adsorbate_atoms)[0]]
        adsorbate_symbols.sort()
        return "".join(adsorbate_symbols)

    def get_adsorbate_sites(self):

        adsorbate_atoms = self.surface.adsorbate_atoms
        return np.where(self.surface.adsorbate_atoms)[0]

    def get_surface_sites(self):
        
        surface_atoms = self.surface.surface_atoms
        return np.where(self.surface.surface_atoms)[0]

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
        for symbol in neighbor_symbols:
            if symbol!='H' and symbol!=self.B:
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

def plot_hexbin(targ, pred, key, title="", scale="linear", 
                inc_factor = 1.1, dec_factor = 0.9,
                bins=None, plot_helper_lines=False,
                cmap='viridis'):
    
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
        gridsize=60,
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

        pred = test_preds[index].numpy()
        targ = test_targs[index].numpy()

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

def get_morgan_parse_parent(cryst):

    while cryst.parentjob.config.name!='morgan_gga_u_parse' and cryst.parentjob.config.name!='pbe_u_paw_spinpol_opt_vasp':
        cryst = cryst.parentjob.parent

    return cryst

def get_crystals_containing_elements(element_list, configs):

    # for a given element list, find crystals with given config that contain these and only these elements

    crystals = Crystal.objects.filter(parentjob__config__in=configs,parentjob__group=group)

    for element in element_list:
        crystals = crystals.filter(stoichiometry__formula__contains=element)

    final_crystal_list=[]
    for cryst in crystals:
        structure = cryst.as_pymatgen_structure()
        crystal_element_list = set([spec.symbol for spec in structure.species])
        if len(crystal_element_list)==len(element_list):
            final_crystal_list.append(cryst)


    return final_crystal_list

def check_stoichiometry(crystal, stoichiometry_dictionary):

    # given a dictionary ({element_symbol:ratio}), determine if a crystal has this stoichiometry

    struct = crystal.as_pymatgen_structure()
    keys = set([spec.symbol for spec in struct.species])

    for key in keys:

        frac = np.sum([spec.symbol==key for spec in struct.species])/len(struct.species)
        if np.abs(frac-stoichiometry_dictionary[key])>0.2/5.0:
            return False

    return True

def get_energy_per_atom(crystal):

    try:
        totalenergy = crystal.calcs.last().props['totalenergy']
    except:
        totalenergy = np.nan
    struct = crystal.as_pymatgen_structure()
    numatoms = len(struct.species)

    return totalenergy/numatoms

def get_crystal_list(dictionary):

    # get list of crystals that have the stoichiometry determined by the dictionary
    
    element_list = list(dictionary.keys())

    # get crystals that have the right elements
    crystal_list = get_crystals_containing_elements(element_list, [opt_config, morgan_parse_config])

    # filter for crystals with the right stoichiometry
    new_crystal_list = []
    for cryst in crystal_list:
        if check_stoichiometry(cryst, dictionary):
            new_crystal_list.append(cryst)

    return new_crystal_list

def print_bulk_info(dictionary):

    new_crystal_list = get_crystal_list(dictionary)

    for cryst in new_crystal_list:
        
        try:
            es_job = cryst.childjobs.get(config__name='pbe_u_paw_spinpol_es_vasp',status='done')
            es_job_id = es_job.id
        except:
            es_job_id = None

        print(cryst.stoichiometry.formula, cryst.id, es_job_id, get_energy_per_atom(cryst))

def normalize(x):

    return x

def value_to_hex(m, x):

    return mcolors.to_hex(m.to_rgba(x))

def get_colors(value_array, vmin=-0.5, vmax=0.2):

    # default vmin and vmax are set for Mn d-band center
    norm = mpl.colors.Normalize(vmin, vmax)
    #cmap = cm.Reds
    #cmap = cm.viridis
    #cmap = cm.YlOrRd

    #make my own colormap from hex list
    cmap = mcolors.ListedColormap(['#000000','#662827', '#cc4f4e',\
        '#ff7371', '#ffb1b0', '#ffefef'])

    #cmap = sns.color_palette("rocket_r", as_cmap=True)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    colors=[value_to_hex(m,value) for value in value_array]
    return colors

def get_site_bandcenters(sa, M, band):

    structure = sa.structure
    symbols = [spec.symbol for spec in structure.species]
    indices = np.where([symbol==M for symbol in symbols])[0]
    return np.array(sa.get_site_bandcenters()[band])[indices]
