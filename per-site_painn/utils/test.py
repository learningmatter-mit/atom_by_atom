
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

morgan = Crystal.objects.get(id=140513600)
print(morgan.as_pymatgen_structure().sites)
#coords = [{} for s in morgan.as_pymatgen_structure().sites]
