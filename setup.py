from setuptools import setup, find_packages

# Read requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='atombyatom',
    version="0.1.0",
    description='atombyatom',
    author='Jaclyn R. Lunger, Jessica Karaguesian, Hoje Chun, Jiayu Peng, Yitong Tseo, Chung Hsuan Shan, Byungchan Han, Yang Shao-Horn, Rafael Gomez-Bombarelli',
    author_email='rafagb@mit.edu',
    packages=find_packages(),
    install_requirements=required,
    package_data={'atombyatom': ['data/*json']},
    url='https://github.com/learningmatter-mit/atom_by_atom',
    entry_points={"console_scripts":["atombyatom=atombyatom.cli.main:main"]},
)
