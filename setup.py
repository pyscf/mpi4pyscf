#!/usr/bin/env python

import os
from setuptools import setup, find_packages

topdir = os.path.abspath(os.path.join(__file__, '..'))

def get_version():
    with open(os.path.join(topdir, 'mpi4pyscf', '__init__.py'), 'r') as f:
        for line in f.readlines():
            if line.startswith('__version__'):
                return eval(line.strip().split(' = ')[1])
    raise ValueError("Version string not found")

with open(os.path.join(topdir, 'README.md'), 'r') as f:
    long_description = f.read()

with open(os.path.join(topdir, 'requirements.txt'), 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='mpi4pyscf',
    version=get_version(),
    description='An MPI plugin for PySCF.',
    long_description=long_description,
    url='http://www.pyscf.org',
    download_url='https://github.com/pyscf/mpi4pyscf',
    author='Qiming Sun',
    author_email='osirpt.sun@gmail.com',
    install_requires=requirements,
    license='GPLv3',
    classifiers=[
'Development Status :: 4 - Beta',
'Intended Audience :: Science/Research',
'Intended Audience :: Developers',
'License :: OSI Approved :: GNU Affero General Public License v3',
'Programming Language :: Python',
'Programming Language :: Python :: 2.7',
'Programming Language :: Python :: 3.4',
'Programming Language :: Python :: 3.5',
'Programming Language :: Python :: 3.6',
'Programming Language :: Python :: 3.7',
'Programming Language :: Python :: 3.8',
'Topic :: Software Development',
'Topic :: Scientific/Engineering',
'Operating System :: POSIX',
'Operating System :: Unix',
],
    packages=find_packages()
)

