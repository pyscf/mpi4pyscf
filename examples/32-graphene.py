#!/usr/bin/env python

import time
import numpy
#from pyscf.pbc import df as pdf
from mpi4pyscf.pbc import df as pdf
from pyscf.pbc import scf as pbchf
from pyscf.pbc import gto as pbcgto

nk = 2
kpts = [nk,nk,1]
Lz = 25 # Smallest Lz value for ~1e-6 convergence in absolute energy
a = 1.42 # bond length in graphene
e = []
t = []
pseudo = 'gth-pade'

##################################################
#
# 2D PBC with AFT
#
##################################################
cell = pbcgto.Cell()
cell.build(unit = 'B',
           a = [[4.6298286730500005, 0.0, 0.0], [-2.3149143365249993, 4.009549246030899, 0.0], [0.0, 0.0, Lz]],
           atom = 'C 0 0 0; C 0 2.67303283 0',
           dimension=2,
           low_dim_ft_type = 'inf_vacuum',
           pseudo = pseudo,
           verbose = 4,
           precision = 1e-6,
           basis='gth-szv')
t0 = time.time()
mf = pbchf.KRHF(cell)
mf.with_df = pdf.AFTDF(cell)
mf.kpts = cell.make_kpts(kpts)
mf.conv_tol = 1e-6
e.append(mf.kernel())
t.append(time.time() - t0)

##################################################
#
# 2D PBC with FFT
#
##################################################
cell = pbcgto.Cell()
cell.build(unit = 'B',
           a = [[4.6298286730500005, 0.0, 0.0], [-2.3149143365249993, 4.009549246030899, 0.0], [0.0, 0.0, Lz]],
           atom = 'C 0 0 0; C 0 2.67303283 0',
           dimension=2,
           pseudo = pseudo,
           verbose = 4,
           precision = 1e-6,
           low_dim_ft_type='analytic_2d_1',
           basis='gth-szv')
t0 = time.time()
mf = pbchf.KRHF(cell)
mf.with_df = pdf.FFTDF(cell)
mf.kpts = cell.make_kpts(kpts)
mf.conv_tol = 1e-6
e.append(mf.kernel())
t.append(time.time() - t0)

##################################################
#
# 2D PBC with GDF
#
##################################################
t0 = time.time()
mf = pbchf.KRHF(cell)
mf.with_df = pdf.GDF(cell)
mf.kpts = cell.make_kpts(kpts)
mf.conv_tol = 1e-6
e.append(mf.kernel())
t.append(time.time() - t0)

##################################################
#
# 2D PBC with MDF
#
##################################################
t0 = time.time()
mf = pbchf.KRHF(cell)
mf.with_df = pdf.MDF(cell)
mf.kpts = cell.make_kpts(kpts)
mf.conv_tol = 1e-6
e.append(mf.kernel())
t.append(time.time() - t0)

print('Energy (AFTDF) (FFTDF) (GDF)   (MDF)')
print(e)
print('Timing (AFTDF) (FFTDF) (GDF)   (MDF)')
print(t)

