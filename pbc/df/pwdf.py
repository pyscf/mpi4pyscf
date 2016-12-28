#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''Density expansion on plane waves'''

import time
import copy
import numpy

from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc.gto import pseudo
from pyscf.pbc.df import pwdf

from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi
from mpi4pyscf.pbc.df import pwdf_jk
#from mpi4pyscf.pbc.df import pwdf_ao2mo

comm = mpi.comm
rank = mpi.rank


@mpi.parallel_call
def get_nuc(mydf, kpts):
    mydf = _sync_mydf(mydf)
# Call the serial code because pw_loop and ft_loop methods are overloaded.
    vne = pwdf.get_nuc(mydf, kpts)
    vne = mpi.reduce(vne)
    return vne

@mpi.parallel_call
def get_pp(mydf, kpts):
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    mydf = _sync_mydf(mydf)
    vpp = pwdf.get_pp_loc_part1(mydf, mydf.cell, kpts_lst)
    vpp = mpi.reduce(lib.asarray(vpp))

    if rank == 0:
        vloc2 = pseudo.pp_int.get_pp_loc_part2(mydf.cell, kpts_lst)
        vppnl = pseudo.pp_int.get_pp_nl(mydf.cell, kpts_lst)
        for k in range(len(kpts_lst)):
            vpp[k] += numpy.asarray(vppnl[k] + vloc2[k], dtype=vpp.dtype)

        if kpts is None or numpy.shape(kpts) == (3,):
            vpp = vpp[0]
        return vpp

def _sync_mydf(mydf):
    mydf.unpack_(comm.bcast(mydf.pack()))
    return mydf


class _PWDF(pwdf.PWDF):
    def __enter__(self):
        return self
    def __exit__(self):
        self.close()
    def close(self):
        self._reg_procs = mpi.del_registry(self._reg_procs)

    def pack(self):
        return {'verbose'   : self.verbose,
                'max_memory': self.max_memory,
                'kpts'      : self.kpts,
                'gs'        : self.gs}
    def unpack_(self, dfdic):
        self.__dict__.update(dfdic)
        return self

    def prange(self, start, stop, step=None):
        # affect pw_loop and ft_loop function
        size = stop - start
        mpi_size = mpi.pool.size
        segsize = (size+mpi_size-1) // mpi_size
        if step is None:
            step = segsize
        else:
            step = min(step, segsize)
        start = min(size, start + rank * segsize)
        stop = min(size, start + segsize)
        return lib.prange(start, stop, step)
    mpi_prange = prange

    get_nuc = get_nuc
    get_pp = get_pp

    def get_jk(self, dm, hermi=1, kpts=None, kpt_band=None,
               with_j=True, with_k=True, exxdiv='ewald'):
        '''Gamma-point calculation by default'''
        if kpts is None:
            if numpy.all(self.kpts == 0):
                kpts = numpy.zeros(3)
            else:
                kpts = self.kpts
        else:
            kpts = numpy.asarray(kpts)

        if kpts.shape == (3,):
            return pwdf_jk.get_jk(self, dm, hermi, kpts, kpt_band, with_j,
                                  with_k, exxdiv)

        vj = vk = None
        if with_k:
            vk = pwdf_jk.get_k_kpts(self, dm, hermi, kpts, kpt_band, exxdiv)
        if with_j:
            vj = pwdf_jk.get_j_kpts(self, dm, hermi, kpts, kpt_band)
        return vj, vk

PWDF = mpi.register_class(_PWDF)


if __name__ == '__main__':
    # run with mpirun -n
    from pyscf.pbc import gto as pgto
    from mpi4pyscf.pbc import df
    cell = pgto.Cell()
    cell.atom = 'He 1. .5 .5; C .1 1.3 2.1'
    cell.basis = {'He': [(0, (2.5, 1)), (0, (1., 1))],
                  'C' :'gth-szv',}
    cell.pseudo = {'C':'gth-pade'}
    cell.h = numpy.eye(3) * 2.5
    cell.gs = [5] * 3
    cell.build()
    numpy.random.seed(19)
    kpts = numpy.random.random((5,3))

    mydf = df.PWDF(cell)
    v = mydf.get_nuc()
    print(v.shape)
    v = mydf.get_pp(kpts)
    print(v.shape)

    cell = pgto.M(atom='He 0 0 0; He 0 0 1', h=numpy.eye(3)*4, gs=[5]*3)
    mydf = df.PWDF(cell)
    nao = cell.nao_nr()
    dm = numpy.ones((nao,nao))
    vj, vk = mydf.get_jk(dm)
    print(vj.shape)
    print(vk.shape)

    dm_kpts = [dm]*5
    vj, vk = mydf.get_jk(dm_kpts, kpts=kpts)
    print(vj.shape)
    print(vk.shape)

    mydf.close()

