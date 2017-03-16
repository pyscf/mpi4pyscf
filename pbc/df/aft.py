#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''Density expansion on plane waves'''

import time
import ctypes
import copy
import numpy

from pyscf import lib
from pyscf import gto
from pyscf.pbc.df import incore
from pyscf.pbc.gto import pseudo
from pyscf.pbc.df import aft
from pyscf.gto.mole import PTR_COORD

from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi
from mpi4pyscf.pbc.df import aft_jk
#from mpi4pyscf.pbc.df import aft_ao2mo

comm = mpi.comm
rank = mpi.rank


@mpi.parallel_call
def get_nuc(mydf, kpts=None):
    mydf = _sync_mydf(mydf)
# Call the serial code because pw_loop and ft_loop methods are overloaded.
    vne = aft.get_nuc(mydf, kpts)
    vne = mpi.reduce(vne)
    return vne

get_pp_loc_part1 = get_nuc

@mpi.parallel_call
def get_pp(mydf, kpts=None):
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    mydf = _sync_mydf(mydf)
    vpp = aft.get_pp_loc_part1(mydf, kpts_lst)
    vpp = mpi.reduce(lib.asarray(vpp))

    if rank == 0:
        vloc2 = pseudo.pp_int.get_pp_loc_part2(mydf.cell, kpts_lst)
        vppnl = pseudo.pp_int.get_pp_nl(mydf.cell, kpts_lst)
        for k in range(len(kpts_lst)):
            vpp[k] += numpy.asarray(vppnl[k] + vloc2[k], dtype=vpp.dtype)

        if kpts is None or numpy.shape(kpts) == (3,):
            vpp = vpp[0]
        return vpp

# Note on each proccessor, _int_nuc_vloc computes only a fraction of the entire vj.
# It is because the summation over real space images are splited by mpi.static_partition
def _int_nuc_vloc(mydf, nuccell, kpts, intor='cint3c2e_sph'):
    '''Vnuc - Vloc'''
    cell = mydf.cell
    rcut = max(cell.rcut, nuccell.rcut)
    Ls = cell.get_lattice_Ls(rcut=rcut)
    expLk = numpy.asarray(numpy.exp(1j*numpy.dot(Ls, kpts.T)), order='C')
    nkpts = len(kpts)

    fakenuc = aft._fake_nuc(cell)
    fakenuc._atm, fakenuc._bas, fakenuc._env = \
            gto.conc_env(nuccell._atm, nuccell._bas, nuccell._env,
                         fakenuc._atm, fakenuc._bas, fakenuc._env)
    charge = cell.atom_charges()
    charge = numpy.append(charge, -charge)  # (charge-of-nuccell, charge-of-fakenuc)

    nao = cell.nao_nr()
    #:buf = [numpy.zeros((nao,nao), order='F', dtype=numpy.complex128)
    #:       for k in range(nkpts)]
    buf = numpy.zeros((nkpts,8,nao,nao),
                      dtype=numpy.complex128).transpose(0,3,2,1)
    ints = incore._wrap_int3c(cell, fakenuc, intor, 1, Ls, buf)
    atm, bas, env = ints._envs[:3]

    xyz = numpy.asarray(cell.atom_coords(), order='C')
    ptr_coordL = atm[:cell.natm,PTR_COORD]
    ptr_coordL = numpy.vstack((ptr_coordL,ptr_coordL+1,ptr_coordL+2)).T.copy('C')
    nuc = 0
    for atm0, atm1 in lib.prange(0, fakenuc.natm, 8):
        c_shls_slice = (ctypes.c_int*6)(0, cell.nbas, cell.nbas, cell.nbas*2,
                                        cell.nbas*2+atm0, cell.nbas*2+atm1)

        for l in mpi.static_partition(range(len(Ls))):
            L1 = Ls[l]
            env[ptr_coordL] = xyz + L1
            exp_Lk = numpy.einsum('k,ik->ik', expLk[l].conj(), expLk[:l+1])
            exp_Lk = numpy.asarray(exp_Lk, order='C')
            exp_Lk[l] = .5
            ints(exp_Lk, c_shls_slice)
        v = buf[:,:,:,:atm1-atm0]
        nuc += numpy.einsum('kijz,z->kij', v, charge[atm0:atm1])
        v[:] = 0

    nuc = nuc + nuc.transpose(0,2,1).conj()
    nuc = lib.pack_tril(nuc)
# nuc is mpi.reduced in get_nuc function
    if rank == 0 and cell.dimension == 3:
        nucbar = sum([z/nuccell.bas_exp(i)[0] for i,z in enumerate(cell.atom_charges())])
        nucbar *= numpy.pi/cell.vol
        ovlp = cell.pbc_intor('cint1e_ovlp_sph', 1, lib.HERMITIAN, kpts)
        for k in range(nkpts):
            s = lib.pack_tril(ovlp[k])
            nuc[k] += nucbar * s
    return nuc


def _sync_mydf(mydf):
    mydf.unpack_(comm.bcast(mydf.pack()))
    return mydf


@mpi.register_class
class AFTDF(aft.AFTDF):

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

    _int_nuc_vloc = _int_nuc_vloc
    get_nuc = get_nuc
    get_pp = get_pp

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
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
            return aft_jk.get_jk(self, dm, hermi, kpts, kpts_band, with_j,
                                  with_k, exxdiv)

        vj = vk = None
        if with_k:
            vk = aft_jk.get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv)
        if with_j:
            vj = aft_jk.get_j_kpts(self, dm, hermi, kpts, kpts_band)
        return vj, vk


if __name__ == '__main__':
    # run with mpirun -n
    from pyscf.pbc import gto as pgto
    from mpi4pyscf.pbc import df
    cell = pgto.Cell()
    cell.atom = 'He 1. .5 .5; C .1 1.3 2.1'
    cell.basis = {'He': [(0, (2.5, 1)), (0, (1., 1))],
                  'C' :'gth-szv',}
    cell.pseudo = {'C':'gth-pade'}
    cell.a = numpy.eye(3) * 2.5
    cell.gs = [5] * 3
    cell.build()
    numpy.random.seed(19)
    kpts = numpy.random.random((5,3))

    mydf = df.AFTDF(cell)
    v = mydf.get_nuc()
    print(v.shape)
    v = mydf.get_pp(kpts)
    print(v.shape)

    cell = pgto.M(atom='He 0 0 0; He 0 0 1', a=numpy.eye(3)*4, gs=[5]*3)
    mydf = df.AFTDF(cell)
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

