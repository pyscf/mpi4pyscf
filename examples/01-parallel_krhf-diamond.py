#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Run the hybrid parallel mode in command line
OMP_NUM_THREADS=2 mpirun -np 2 python 01-parallel_krhf-diamond.py
'''

import pyscf.pbc.gto as pbcgto
import pyscf.pbc.scf as pscf
from mpi4pyscf.pbc import df
#from pyscf.pbc import df

cell = pbcgto.Cell()
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.gs = [7]*3
cell.verbose = 5
cell.build()

kpts = cell.make_kpts([2]*3)
mf = pscf.KRHF(cell, kpts)
mf.with_df = df.FFTDF(cell, kpts)
mf.kernel()
