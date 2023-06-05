# setup.py file for installing the package

from setuptools import setup, find_packages

setup(
    name='atombyatom',
    version='0.1.0',
    description='atombyatom',
    author='Jaclyn R. Lunger, Jessica Karaguesian, Hoje Chun, Jiayu Peng, Yitong Tseo, Chung Hsuan Shan, Byungchan Han, Yang Shao-Horn, Rafael Gomez-Bombarelli',
    author_email='rafagb@mit.edu',
    packages=find_packages(),
    package_data=  {'atombyatom': ['data/*json']},
    url='https://github.com/learningmatter-mit/atom_by_atom',
)
