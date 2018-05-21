#!/usr/bin/env python

'''
mpirun -np 2 python 04-parallel_ccsd.py
'''

from pyscf import gto
from pyscf import scf
from mpi4pyscf import cc as mpicc
from pyscf import cc as serial_cc


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

mycc = mpicc.RCCSD(mf)
mycc.diis_file = 'mpi_ccdiis.h5'
mycc.kernel()

mycc.restore_from_diis_('mpi_ccdiis.h5')
mycc.kernel()


s_cc = serial_cc.RCCSD(mf)
s_cc.diis_file = 'serial_ccdiis.h5'
s_cc.kernel()

p_cc = mpicc.RCCSD(mf)
p_cc.restore_from_diis_('serial_ccdiis.h5')
p_cc.kernel()
