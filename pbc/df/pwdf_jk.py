#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
JK with analytic Fourier transformation
'''

import time
import numpy
from pyscf.pbc.df import pwdf_jk
from mpi4pyscf.tools import mpi

comm = mpi.comm
rank = mpi.rank


get_j_kpts = mpi.call_then_reduce(pwdf_jk.get_j_kpts)

@mpi.parallel_call
def get_k_kpts(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), kpt_band=None,
               exxdiv=None):
    if rank != 0:  # to apply df_jk._ewald_exxdiv_for_G0 function once
        exxdiv = None
    vk = pwdf_jk.get_k_kpts(mydf, dm_kpts, hermi, kpts, kpt_band, exxdiv)
    vk = mpi.reduce(vk)
    return vk


##################################################
#
# Single k-point
#
##################################################

@mpi.parallel_call
def get_jk(mydf, dm, hermi=1, kpt=numpy.zeros(3),
           kpt_band=None, with_j=True, with_k=True, exxdiv=None):
    '''JK for given k-point'''
    if rank != 0:  # to apply df_jk._ewald_exxdiv_for_G0 function once
        exxdiv = None
    vj, vk = pwdf_jk.get_jk(mydf, dm, hermi, kpt, kpt_band, with_j, with_k, exxdiv)

    if with_j: vj = mpi.reduce(vj)
    if with_k: vk = mpi.reduce(vk)
    return vj, vk


if __name__ == '__main__':
    from pyscf.pbc import gto as pgto
    from pyscf.pbc import scf as pscf
    from pyscf.pbc.df import pwdf

    L = 5.
    n = 5
    cell = pgto.Cell()
    cell.a = numpy.diag([L,L,L])
    cell.gs = numpy.array([n,n,n])

    cell.atom = '''He    3.    2.       3.
                   He    1.    1.       1.'''
    #cell.basis = {'He': [[0, (1.0, 1.0)]]}
    #cell.basis = '631g'
    #cell.basis = {'He': [[0, (2.4, 1)], [1, (1.1, 1)]]}
    cell.basis = 'ccpvdz'
    cell.verbose = 0
    cell.build(0,0)
    cell.verbose = 5

    df = pwdf.PWDF(cell)
    df.gs = (15,)*3
    dm = pscf.RHF(cell).get_init_guess()
    vj, vk = df.get_jk(cell, dm)
    print(numpy.einsum('ij,ji->', df.get_nuc(cell), dm), 'ref=-10.384051732669329')
    df.analytic_ft = True
    #print(numpy.einsum('ij,ji->', vj, dm), 'ref=5.3766911667862516')
    #print(numpy.einsum('ij,ji->', vk, dm), 'ref=8.2255177602309022')
    print(numpy.einsum('ij,ji->', df.get_nuc(cell), dm), 'ref=-10.447018516011319')

