#!/usr/bin/env python

'''
mpirun -np 2 python 04-parallel_mol_hf.py
'''

import numpy
from pyscf import gto, scf
from mpi4pyscf import scf as mpi_scf

mol = gto.M(atom='''O    0.   0.       0.
                    H    0.   -0.757   0.587
                    H    0.   0.757    0.587''',
            basis='cc-pvtz')

mf = mpi_scf.UHF(mol)
mf.direct_scf_tol = 1e-9
mf.kernel()

mf = mpi_scf.RHF(mol)
mf.kernel()

mol = gto.M(atom='H 0 -6.5 0; H 0 .5 4; H -1.1 0.2 0.2; H 0.6 0.5 0.4',
            basis='cc-pvtz')

nao = mol.nao
numpy.random.seed(1)
dm = numpy.random.random((2,nao,nao))

mf = mpi_scf.RHF(mol)
mf.direct_scf_tol = 1e-9

vj, vk = mf.get_jk(mol, dm, hermi=0)
vj0, vk0 = scf.hf.get_jk(mol, dm, hermi=0)
print(abs(vj0-vj).max())
print(abs(vk0-vk).max())
vj = mf.get_j(mol, dm, hermi=0)
vk = mf.get_k(mol, dm, hermi=0)
print(abs(vj0-vj).max())
print(abs(vk0-vk).max())

dm = dm + dm.transpose(0,2,1)
vj, vk = mpi_scf.hf.get_jk(mol, dm, hermi=1)
vj0, vk0 = scf.hf.get_jk(mol, dm)
print(abs(vj0-vj).max())
print(abs(vk0-vk).max())
vj = mpi_scf.hf.get_j(mol, dm, hermi=1)
vk = mpi_scf.hf.get_k(mol, dm, hermi=1)
print(abs(vj0-vj).max())
print(abs(vk0-vk).max())
