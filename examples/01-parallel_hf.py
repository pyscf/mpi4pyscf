#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Parallelize gamma point
'''

import numpy
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.scf as pscf
from mpi4pyscf.pbc import df as mpidf

cell = pbcgto.Cell()
cell.atom = [['C', ([ 0.,  0.,  0.])],
             ['C', ([ 0.8917,  0.8917,  0.8917])],
             ['C', ([ 1.7834,  1.7834,  0.    ])],
             ['C', ([ 2.6751,  2.6751,  0.8917])],
             ['C', ([ 1.7834,  0.    ,  1.7834])],
             ['C', ([ 2.6751,  0.8917,  2.6751])],
             ['C', ([ 0.    ,  1.7834,  1.7834])],
             ['C', ([ 0.8917,  2.6751,  2.6751])]
            ]
cell.a = numpy.eye(3) * 3.5668
cell.basis = 'sto3g'
cell.mesh = [10] * 3
cell.verbose = 4
cell.build()
cell.max_memory = 1

mydf = mpidf.MDF(cell)
mydf.auxbasis = 'weigend'

mf = pscf.RHF(cell)
mf.exxdiv = 'ewald'
mf.with_df = mydf
mf.kernel()


mydf = mpidf.DF(cell)
mydf.auxbasis = 'weigend'
mydf.mesh = [5] * 3

mf = pscf.RHF(cell)
mf.exxdiv = 'ewald'
mf.with_df = mydf
mf.kernel() # -299.323528879552


mydf = mpidf.MDF(cell)
mydf.auxbasis = 'weigend'
mydf.mesh = [5] * 3

mf = pscf.RHF(cell)
mf.exxdiv = 'ewald'
mf.with_df = mydf
mf.kernel() # -299.328386756269
