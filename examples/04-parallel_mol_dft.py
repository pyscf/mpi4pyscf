#!/usr/bin/env python

'''
mpirun -np 2 python 04-parallel_mol_dft.py
'''

import numpy
from pyscf import gto, dft
from mpi4pyscf import dft as mpi_dft

mol = gto.M(atom='O 0 .5 0; H -1.1 0.2 0.2; H 0.6 0.5 0.4',
            basis='cc-pvtz')
nao = mol.nao
numpy.random.seed(1)
dm = numpy.random.random((nao,nao))
dm = dm + dm.T

#
# RKS
#
mf = mpi_dft.RKS(mol)
mf.xc = 'b3lyp'
vxc = mf.get_veff(mol, dm)
mf.run()

mf0 = dft.RKS(mol)
mf0.xc = 'b3lyp'
vxc0 = mf0.get_veff(mol, dm)
mf0.run()
print(abs(vxc0-vxc).max())

#
# UKS
#
dm = numpy.random.random((2,nao,nao))
dm = dm + dm.transpose(0,2,1)
mol.spin = 2
mf = mpi_dft.UKS(mol)
mf.xc = 'b3lyp'
vxc = mf.get_veff(mol, dm)
mf.run()

mf0 = dft.UKS(mol)
mf0.xc = 'b3lyp'
vxc0 = mf0.get_veff(mol, dm)
mf0.run()
print(abs(vxc0-vxc).max())
