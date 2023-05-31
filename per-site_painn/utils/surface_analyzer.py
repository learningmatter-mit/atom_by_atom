import sys
import os
import django

import matplotlib.pyplot as plt
from scipy.stats import norm

sys.path.append('/home/lungerja/projects/htvs/')
sys.path.append('/home/lungerja/projects/htvs/djangochem/')

# setup the djhango settings file.  Change this to use the settings file that connects you to your desired database
os.environ["DJANGO_SETTINGS_MODULE"] = "djangochem.settings.orgel"
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

# this must be run to setup access to the django settings and make database access work etc.
django.setup()

from jobs.models import Job, JobConfig
from django.contrib.contenttypes.models import ContentType
from pgmols.models import *
from docking.models import * 

from structure import *
from dos import *

class BulkAnalyzer():

    def __init__(
            self,
            id_,
    ):

        # basic information
        self.bulk = Crystal.objects.get(id=id_)
        self.structure = self.bulk.as_pymatgen_structure()
        self.pdos = self.get_pdos_object()

    def get_pdos_object(self):

        childjobs = self.bulk.childjobs.filter(config__name='pbe_u_paw_spinpol_es_vasp', status='done')

        if childjobs:
            pdos = ProjectedDOS.objects.filter(parentjob=childjobs[0])
            if pdos:
                return pdos[0]
            else:
                return None

        return None

    def get_bandcenter(self, site):

        return get_site_valence_bandcenter(self.pdos, site)

    def get_bandwidth(self, site):

        return get_site_valence_bandwidth(self.pdos, site)

    def get_structure_with_site_properties(self):

        # return a structure with all the proper site properties
        bandcenters = [self.get_bandcenter(site) for site in range(len(self.structure.sites))]
        bandwidths = [self.get_bandwidth(site) for site in range(len(self.structure.sites))]

        site_properties = {'bandcenter':bandcenters, 'bandwidth':bandwidths}
        return self.structure.copy(site_properties=site_properties)

class SurfaceAnalyzer():

    '''
    analyzer class that takes an id of an optimized clean surface and:
    generates a structure with 'magmoms', 'badfilling', 'bandcenters', 'bandwidths' in the site properties
    '''

    def __init__(
            self,
            id_,  
    ):
        
        # basic information
        self.surface = Surface.objects.get(id=id_)
        self.structure = self.surface.as_pymatgen_structure()
        self.pdos = self.get_pdos_object()
    
    def get_pdos_object(self):

        childjobs = self.surface.childjobs.filter(config__name='pbe_u_paw_spinpol_es_surf_vasp', status='done')
        
        if childjobs:
            pdos = ProjectedDOS.objects.filter(parentjob=childjobs[0])
            if pdos:
                return pdos[0]
            else:
                return None
       
        return None

    def get_baders(self): 

        bader_charges = self.surface.calcs.filter(props__atomiccharges__isnull=False)
        if bader_charges.exists():
            return bader_charges.values_list('props__atomiccharges',flat=True)[0]['bader']
        return None
    
    def get_bandfilling(self, site):

        # for a given site, get the number of valence electrons minus the bader charge (total electrons in valence shell)
        
        bader = self.get_baders()
        el = Element(list(self.structure.sites[site].species.as_dict().keys())[0])
        otype = orbital_type(el)

        if el.block == 'f':
            return neutral_valence(el)
        else:
            return abs(neutral_valence(el)-bader[site])

    def get_bandcenter(self, site):
        
        return get_site_valence_bandcenter(self.pdos, site)
            
    def get_bandwidth(self, site):
        
        return get_site_valence_bandwidth(self.pdos, site)

    def get_structure_with_site_properties(self):

        # return a structure with all the proper site properties
        bandfilling = [self.get_bandfilling(site) for site in range(len(self.structure.sites))]
        bandcenters = [self.get_bandcenter(site) for site in range(len(self.structure.sites))]
        bandwidths = [self.get_bandwidth(site) for site in range(len(self.structure.sites))]

        site_properties = {'bandfilling':bandfilling, 'bandcenter':bandcenters, 'bandwidth':bandwidths}
        return self.structure.copy(site_properties=site_properties)
    
    def plot_site_valence_dos(self, site, ax, color):
       
        # given a site, plot out the valence dos (both the actual dos and the estimate using per-site properties)
        energies, densities = get_site_valence_dos(self.get_pdos_object(), site)
        ax.plot(energies, densities, color=color, linestyle='-', linewidth=2)
        
        mu = self.get_bandcenter(site)
        sigma = self.get_bandwidth(site)
        Nv = self.get_bandfilling(site)

        ax.plot(energies, Nv*norm.pdf(energies, mu, sigma), color=color, linestyle='--', linewidth=2)

        

class SurfaceWAdsorbateAnalyzer(SurfaceAnalyzer):
    '''
    analyzer class that takes an id of an optimized surface with adsorbate and:
    1) identifies the adsorbate, if it's in a top site, and if this surface is conducting
    2) generates structure with 'magmoms', 'bader', 'bandcenters', 'bandvariances', bandkurtosis' in the site properties (if available) 
    '''
    def __init__(
            self,
            id_,  
    ):
        
        # basic information
        self.surface = Surface.objects.get(id=id_)
        self.structure = self.surface.as_pymatgen_structure()
        self.pdos = self.get_pdos_object()
        self.adsorbate = get_adsorbate(self.surface)
        self.miller_index = self.surface.miller_index.hkl
        

    def get_opt_clean(self):

        be = BindingEnergy.objects.get(surface_w_adsorbate=self.surface)
        return be.clean_surface

    @property
    def conducting(self):
        
        return is_conducting(self.pdos)

    @property
    def top_site(self):
       
        return is_top_site(self.surface)

    @property
    def f_block_active_site(self):
        
        active_element = get_active_species(self.surface)
        if active_element:
            return get_active_species(self.surface).block == 'f'
        else:
            return 'unknown'
    

    @property
    def binding_energy(self):

        be = BindingEnergy.objects.get(surface_w_adsorbate=self.surface)
        return be.value

    @property
    def descriptor1(self):

        # bandcenter of adsorbed oxygen
        O_adsorbate_site = get_O_adsorbate_sites(self.surface)[0]
        return self.get_bandcenter(O_adsorbate_site)
   
    @property
    def descriptor2(self):

        # bandcenter of the active site
        active_site = get_active_site(self.surface)
        return self.get_bandcenter(active_site)

    @property
    def descriptor3(self):

        # bandcenter of adsorbed oxygen minus bandcenter of active site
        return self.descriptor1 - self.descriptor2

    @property
    def descriptor4(self):

        # midway point between adsorbed oxygen and active site bandcenters (energy in the bond?)
        return 0.5*(self.descriptor1+self.descriptor2)

    @property
    def descriptor5(self):

        # average energy of the overlapping valence dos between active site and adsorbed oxygen
        # valence dos is estimated using a normal distribution and the filling, center and width 
        energies = np.arange(-10,2,0.1)

        active_site = get_active_site(self.surface)
        mu = self.get_bandcenter(active_site)
        sigma = self.get_bandwidth(active_site)
        Nv = self.get_bandfilling(active_site)
        active_site_dos = Nv*norm.pdf(energies, mu, sigma)

        adsorbed_O_site = get_O_adsorbate_sites(self.surface)[0]
        mu = self.get_bandcenter(adsorbed_O_site)
        sigma = self.get_bandwidth(adsorbed_O_site)
        Nv = self.get_bandfilling(adsorbed_O_site)
        adsorbed_O_site_dos = Nv*norm.pdf(energies, mu, sigma)

        min_function = [np.min([active_site_dos[i],adsorbed_O_site_dos[i]]) for i in range(len(active_site_dos))]
        normalization = integrate.simps(min_function, energies)
        center = integrate.simps(min_function*energies, energies)
        return center/normalization
