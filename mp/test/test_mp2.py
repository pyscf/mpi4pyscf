#!/usr/bin/env python
# mpirun -np 2 python test_mp2.py

import numpy
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import mp
from mpi4pyscf import mp as mpimp

mol = gto.Mole()
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]
mol.basis = 'cc-pvdz'
mol.build()
mf = scf.RHF(mol).run()
e = mp.RMP2(mf).kernel()[0]

pt = mpimp.RMP2(mf)
emp2 = pt.kernel()[0]
print(emp2 - e)

