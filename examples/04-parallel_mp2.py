#!/usr/bin/env python

'''
mpirun -np 2 python parallel_mp2.py
'''

from pyscf import gto
from pyscf import scf
#from pyscf import mp
from mpi4pyscf import mp


mol = gto.Mole()
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]
mol.basis = 'cc-pvdz'
mol.verbose = 4
mol.build()
mf = scf.RHF(mol).run()

mp2 = mp.RMP2(mf)
mp2.kernel()
