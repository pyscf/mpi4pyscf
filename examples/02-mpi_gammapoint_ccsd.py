#!/usr/bin/env python

'''
Gamma point post-HF calculation
'''

import numpy
from pyscf.pbc import gto, scf
from pyscf import cc
from mpi4pyscf.pbc import df as mpidf

cell = gto.M(
    a = numpy.eye(3)*3.5668,
    atom = '''H     0.      0.      0.    
              H     0.8917  0.8917  0.8917
              H     1.7834  1.7834  0.    
              H     2.6751  2.6751  0.8917
              H     1.7834  0.      1.7834
              H     2.6751  0.8917  2.6751
              H     0.      1.7834  1.7834
              H     0.8917  2.6751  2.6751''',
    basis = 'sto3g',
    mesh = [15]*3,
    verbose = 4,
)

mf = scf.RHF(cell)
mf.with_df = mpidf.FFTDF(cell)
mf.kernel()
mycc = cc.CCSD(mf)
mycc.kernel()

mf.with_df = mpidf.AFTDF(cell)
mf.with_df.mesh = [10]*3
mf.kernel()
mycc = cc.CCSD(mf)
mycc.kernel()

mf.with_df = mpidf.DF(cell)
mf.with_df.mesh = [7]*3
mf.kernel()
mycc = cc.CCSD(mf)
mycc.kernel()

mf.with_df = mpidf.MDF(cell)
mf.with_df.mesh = [7]*3
mf.kernel()
mycc = cc.CCSD(mf)
mycc.kernel()

