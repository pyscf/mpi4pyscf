#!/usr/bin/env python

'''
mpirun -np 2 python 04-parallel_mol_dft.py
'''

import numpy
from pyscf import gto, dft
from mpi4pyscf.dft import rks as mpi_rks

mol = gto.M(atom='H 0 -6.5 0; H 0 .5 4; H -1.1 0.2 0.2; H 0.6 0.5 0.4',
            basis='cc-pvtz')
nao = mol.nao
numpy.random.seed(1)
dm = numpy.random.random((nao,nao))
dm = dm + dm.T

mf = mpi_rks.RKS(mol)
mf.xc = 'b3lyp'
vxc = mf.get_veff(mol, dm)

mf0 = dft.RKS(mol)
mf0.xc = 'b3lyp'
vxc0 = mf0.get_veff(mol, dm)

print(abs(vxc0-vxc).max())
