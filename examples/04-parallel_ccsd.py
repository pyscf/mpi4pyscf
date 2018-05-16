#!/usr/bin/env python

'''
mpirun -np 2 python parallel_ccsd.py
'''

from pyscf import gto
from pyscf import scf
#from pyscf import cc
from mpi4pyscf import cc


mol = gto.Mole()
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]
mol.basis = '6-31g'
mol.verbose = 4
mol.build()
mf = scf.RHF(mol)
mf.chkfile = 'h2o.chk'
mf.run()

mycc = cc.RCCSD(mf)
mycc.diis_file = 'mpi_ccdiis.h5'
mycc.kernel()

mycc.restore_from_diis_('mpi_ccdiis.h5')
mycc.kernel()

