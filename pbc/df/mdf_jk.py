#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Exact density fitting with Gaussian and planewaves
Ref:
'''

import time
import numpy
from pyscf.pbc.df import mdf_jk
from mpi4pyscf.tools import mpi

comm = mpi.comm
rank = mpi.rank

#
# Divide the Coulomb potential to two parts.  Computing short range part in
# real space, long range part in reciprocal space.
#

def density_fit(mf, auxbasis=None, gs=None, with_df=None):
    '''Generte density-fitting SCF object

    Args:
        auxbasis : str or basis dict
            Same format to the input attribute mol.basis.
            The default basis 'weigend+etb' means weigend-coulomb-fit basis
            for light elements and even-tempered basis for heavy elements.
        gs : tuple
            number of grids in each (+)direction
        with_df : MDF object
    '''
    from mpi4pyscf.pbc.df import mdf
    if with_df is None:
        if hasattr(mf, 'kpts'):
            kpts = mf.kpts
        else:
            kpts = numpy.reshape(mf.kpt, (1,3))
        with_df = mdf.MDF(mf.cell, kpts)
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose
        with_df.auxbasis = auxbasis
        if gs is not None:
            with_df.gs = gs

    mf.with_df = with_df
    return mf


get_j_kpts = mpi.call_then_reduce(mdf_jk.get_j_kpts)

@mpi.parallel_call
def get_k_kpts(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)),
               kpts_band=None, exxdiv=None):
    if rank != 0:  # to apply df_jk._ewald_exxdiv_for_G0 function once
        exxdiv = None
    vk = mdf_jk.get_k_kpts(mydf, dm_kpts, hermi, kpts, kpts_band, exxdiv)
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
    if rank != 0:
        exxdiv = None
    vj, vk = mdf_jk.get_jk(mydf, dm, hermi, kpt, kpt_band, with_j, with_k, exxdiv)

    if with_j: vj = mpi.reduce(vj)
    if with_k: vk = mpi.reduce(vk)
    return vj, vk


if __name__ == '__main__':
    import pyscf.pbc.gto as pgto
    import pyscf.pbc.scf as pscf
    import pyscf.pbc.dft as pdft

    L = 5.
    n = 5
    cell = pgto.Cell()
    cell.a = numpy.diag([L,L,L])
    cell.gs = numpy.array([n,n,n])

    cell.atom = '''C    3.    2.       3.
                   C    1.    1.       1.'''
    #cell.basis = {'He': [[0, (1.0, 1.0)]]}
    #cell.basis = '631g'
    #cell.basis = {'He': [[0, (2.4, 1)], [1, (1.1, 1)]]}
    cell.basis = 'ccpvdz'
    cell.verbose = 0
    cell.build(0,0)
    cell.verbose = 5

    mf = pscf.RHF(cell)
    auxbasis = 'weigend'
    mf = density_fit(mf, auxbasis)
    mf.with_df.gs = (5,) * 3
    dm = mf.get_init_guess()
    vj = mf.get_j(cell, dm)
    print(numpy.einsum('ij,ji->', vj, dm), 'ref=46.698951141791')
    vj, vk = mf.with_df.get_jk(dm)
    print(numpy.einsum('ij,ji->', vj, dm), 'ref=46.698951141791')
    print(numpy.einsum('ij,ji->', vk, dm), 'ref=37.348980782463')

    kpts = cell.make_kpts([2]*3)[:4]
    from mpi4pyscf.pbc.df import MDF
    with_df = MDF(cell, kpts)
    with_df.auxbasis = 'weigend'
    with_df.gs = [5] * 3
    dms = numpy.array([dm]*len(kpts))
    vj, vk = with_df.get_jk(dms, exxdiv=mf.exxdiv, kpts=kpts)
    print(numpy.einsum('ij,ji->', vj[0], dms[0]), - 46.69784775484953)
    print(numpy.einsum('ij,ji->', vj[1], dms[1]), - 46.69815612398016)
    print(numpy.einsum('ij,ji->', vj[2], dms[2]), - 46.69526857884274)
    print(numpy.einsum('ij,ji->', vj[3], dms[3]), - 46.69571387135913)
    print(numpy.einsum('ij,ji->', vk[0], dms[0]), - 37.27054185436846)
    print(numpy.einsum('ij,ji->', vk[1], dms[1]), - 37.27081050772260)
    print(numpy.einsum('ij,ji->', vk[2], dms[2]), - 37.27081024429774)
    print(numpy.einsum('ij,ji->', vk[3], dms[3]), - 37.27090527533854)
