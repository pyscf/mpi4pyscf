#!/usr/bin/env python

'''
mpirun -np 2 python 04-parallel_mol_hf.py
'''

import numpy
from pyscf import gto, scf
from mpi4pyscf.scf import hf as mpi_hf

mol = gto.M(atom='H 0 -6.5 0; H 0 .5 4; H -1.1 0.2 0.2; H 0.6 0.5 0.4',
            basis='cc-pvtz')
nao = mol.nao
numpy.random.seed(1)
dm = numpy.random.random((nao,nao))

mf = mpi_hf.RHF(mol)
mf.direct_scf_tol = 1e-9

vj, vk = mf.get_jk(mol, dm, hermi=0)
vj0, vk0 = scf.hf.get_jk(mol, dm, hermi=0)
print(abs(vj0-vj).max())
print(abs(vk0-vk).max())

dm = dm + dm.T
vj, vk = mpi_hf.get_jk(mol, dm, hermi=1)
vj0, vk0 = scf.hf.get_jk(mol, dm)
print(abs(vj0-vj).max())
print(abs(vk0-vk).max())
