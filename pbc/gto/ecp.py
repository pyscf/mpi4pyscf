#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Short range part of ECP under PBC
'''

import ctypes
import copy
import numpy

from pyscf import lib
from pyscf import gto
#from pyscf.gto import AS_ECPBAS_OFFSET, AS_NECPBAS
from pyscf.gto import PTR_ECPBAS_OFFSET, PTR_NECPBAS

from pyscf.pbc.df import incore
from pyscf.pbc import gto as pgto

from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi

comm = mpi.comm
rank = mpi.rank


@mpi.parallel_call
def ecp_int(cell, kpts=None):
    if rank == 0:
        comm.bcast(cell.dumps())
    else:
        cell = pgto.loads(comm.bcast(None))

    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    ecpcell = gto.Mole()
    ecpcell._atm = cell._atm
    # append a fictitious s function to mimic the auxiliary index in pbc.incore.
    # ptr2last_env_idx to force PBCnr3c_fill_* function to copy the entire "env"
    ptr2last_env_idx = len(cell._env) - 1
    ecpbas = numpy.vstack([[0, 0, 1, 1, 0, ptr2last_env_idx, 0, 0],
                           cell._ecpbas]).astype(numpy.int32)
    ecpcell._bas = ecpbas
    ecpcell._env = cell._env
    # In pbc.incore _ecpbas is appended to two sets of cell._bas and the
    # fictitious s function.
    cell._env[PTR_ECPBAS_OFFSET] = cell.nbas * 2 + 1
    cell._env[PTR_NECPBAS] = len(cell._ecpbas)

    kptij_lst = numpy.hstack((kpts_lst,kpts_lst)).reshape(-1,2,3)
    nkpts = len(kpts_lst)
    if abs(kpts_lst).sum() < 1e-9:  # gamma_point
        dtype = numpy.double
    else:
        dtype = numpy.complex128
    ao_loc = cell.ao_loc_nr()
    nao = ao_loc[-1]
    mat = numpy.zeros((nkpts,nao,nao), dtype=dtype)

    intor = cell._add_suffix('ECPscalar')
    int3c = incore.wrap_int3c(cell, ecpcell, intor, kptij_lst=kptij_lst)

    # shls_slice of auxiliary index (0,1) corresponds to the fictitious s function
    tasks = [(i, i+1, j, j+1, 0, 1) # shls_slice
             for i in range(cell.nbas) for j in range(i+1)]
    for shls_slice in mpi.work_stealing_partition(tasks):
        i0 = ao_loc[shls_slice[0]]
        i1 = ao_loc[shls_slice[1]]
        j0 = ao_loc[shls_slice[2]]
        j1 = ao_loc[shls_slice[3]]
        buf = numpy.empty((nkpts,i1-i0,j1-j0), dtype=dtype)
        mat[:,i0:i1,j0:j1] = int3c(shls_slice, buf)

    buf = mpi.reduce(mat)
    if rank == 0:
        mat = []
        for k, kpt in enumerate(kpts_lst):
            v = lib.unpack_tril(lib.pack_tril(buf[k]), lib.HERMITIAN)
            if abs(kpt).sum() < 1e-9:  # gamma_point:
                v = v.real
            mat.append(v)
        if kpts is None or numpy.shape(kpts) == (3,):
            mat = mat[0]
        return mat

