#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Parallelized k-points RHF calculation

Run the hybrid parallel mode in command line
OMP_NUM_THREADS=4 mpirun -np 4 python 01-parallel_krhf.py
'''

import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.scf as pscf
from ase.lattice.cubic import Diamond
from ase.lattice import bulk

import time
from pyscf import lib
lib.logger.TIMER_LEVEL = 0

#
# mpi4pyscf provides the same code layout as pyscf
#
#from pyscf.pbc import df
from mpi4pyscf.pbc import df

ase_atom = Diamond(symbol='C', latticeconstant=3.5668)
#ase_atom = bulk('C', 'diamond', a=3.5668)

cell = pbcgto.Cell()
cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
cell.a = ase_atom.cell
cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-pade'
#cell.mesh = [15]*3
#cell.verbose = 4
cell.build()

kpts = cell.make_kpts([2,2,2])
mf = pscf.KRHF(cell, kpts)
mf.with_df = df.FFTDF(cell, kpts)
mf.with_df.occ = True
mf.verbose = 4

ehf = mf.kernel()
print("HF energy (per unit cell) = %.17g" % ehf)
#
# Replace the serial DF module with MPI implementation to parallelize the J/K
# evaluation.
#
Jtime=time.time()
mf.with_df = df.DF(cell, kpts)
#print(mf.scf())
print "Took this long for intg: ", time.time()-Jtime
#
# Similar replacement can be placed on MDF module
#
mf.with_df = df.MDF(cell, kpts)
#print mf.scf()

