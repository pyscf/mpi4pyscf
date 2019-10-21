#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''Density expansion on plane waves'''


import sys
import copy
import numpy
import h5py

from pyscf import lib
from pyscf import gto
from pyscf import dft
from pyscf.pbc import tools
from pyscf.pbc import gto as pgto
from pyscf.pbc.df import ft_ao
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import numint
from pyscf.pbc.gto import pseudo
from pyscf.pbc.df import fft

from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi
from mpi4pyscf.pbc.df import fft_jk as mpi_fft_jk
from mpi4pyscf.pbc.df import fft_occk as mpi_fft_occk

comm = mpi.comm
rank = mpi.rank


@mpi.parallel_call
def get_nuc(mydf, kpts):
    mydf = _sync_mydf(mydf)
    cell = mydf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))
    if abs(kpts_lst).sum() < 1e-9:  # gamma_point
        dtype = numpy.float64
    else:
        dtype = numpy.complex128

    mesh = mydf.mesh
    charge = -cell.atom_charges()
    Gv = cell.get_Gv(mesh)
    SI = cell.get_SI(Gv)
    rhoG = numpy.dot(charge, SI)

    coulG = tools.get_coulG(cell, mesh=mesh, Gv=Gv)
    vneG = rhoG * coulG
    vneR = tools.ifft(vneG, mydf.mesh).real

    vne = [0] * len(kpts_lst)
    for ao_ks_etc, p0, p1 in mydf.mpi_aoR_loop(mydf.grids, kpts_lst):
        ao_ks = ao_ks_etc[0]
        for k, ao in enumerate(ao_ks):
            vne[k] += lib.dot(ao.T.conj()*vneR[p0:p1], ao)
        ao = ao_ks = None
    vne = mpi.reduce(lib.asarray(vne))

    if rank == 0:
        if kpts is None or numpy.shape(kpts) == (3,):
            vne = vne[0]
        return vne


@mpi.parallel_call
def get_pp(mydf, kpts=None):
    mydf = _sync_mydf(mydf)
    cell = mydf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))
    if abs(kpts_lst).sum() < 1e-9:
        dtype = numpy.float64
    else:
        dtype = numpy.complex128

    mesh = mydf.mesh
    SI = cell.get_SI()
    Gv = cell.get_Gv(mesh)
    vpplocG = pseudo.get_vlocG(cell, Gv)
    vpplocG = -numpy.einsum('ij,ij->j', SI, vpplocG)
    ngrids = len(vpplocG)
    nao = cell.nao_nr()
    nkpts = len(kpts_lst)

    # vpploc evaluated in real-space
    vpplocR = tools.ifft(vpplocG, mesh).real
    vpp = numpy.zeros((nkpts,nao,nao), dtype=dtype)
    for ao_ks_etc, p0, p1 in mydf.mpi_aoR_loop(mydf.grids, kpts_lst):
        ao_ks = ao_ks_etc[0]
        for k, ao in enumerate(ao_ks):
            vpp[k] += lib.dot(ao.T.conj()*vpplocR[p0:p1], ao)
        ao = ao_ks = None
    vpp = mpi.reduce(lib.asarray(vpp))

    # vppnonloc evaluated in reciprocal space
    fakemol = gto.Mole()
    fakemol._atm = numpy.zeros((1,gto.ATM_SLOTS), dtype=numpy.int32)
    fakemol._bas = numpy.zeros((1,gto.BAS_SLOTS), dtype=numpy.int32)
    ptr = gto.PTR_ENV_START
    fakemol._env = numpy.zeros(ptr+10)
    fakemol._bas[0,gto.NPRIM_OF ] = 1
    fakemol._bas[0,gto.NCTR_OF  ] = 1
    fakemol._bas[0,gto.PTR_EXP  ] = ptr+3
    fakemol._bas[0,gto.PTR_COEFF] = ptr+4

    # buf for SPG_lmi upto l=0..3 and nl=3
    buf = numpy.empty((48,ngrids), dtype=numpy.complex128)
    def vppnl_by_k(kpt):
        Gk = Gv + kpt
        G_rad = lib.norm(Gk, axis=1)
        aokG = ft_ao.ft_ao(cell, Gv, kpt=kpt) * (ngrids/cell.vol)
        vppnl = 0
        for ia in range(cell.natm):
            symb = cell.atom_symbol(ia)
            if symb not in cell._pseudo:
                continue
            pp = cell._pseudo[symb]
            p1 = 0
            for l, proj in enumerate(pp[5:]):
                rl, nl, hl = proj
                if nl > 0:
                    fakemol._bas[0,gto.ANG_OF] = l
                    fakemol._env[ptr+3] = .5*rl**2
                    fakemol._env[ptr+4] = rl**(l+1.5)*numpy.pi**1.25
                    pYlm_part = dft.numint.eval_ao(fakemol, Gk, deriv=0)

                    p0, p1 = p1, p1+nl*(l*2+1)
                    # pYlm is real, SI[ia] is complex
                    pYlm = numpy.ndarray((nl,l*2+1,ngrids), dtype=numpy.complex128, buffer=buf[p0:p1])
                    for k in range(nl):
                        qkl = pseudo.pp._qli(G_rad*rl, l, k)
                        pYlm[k] = pYlm_part.T * qkl
                    #:SPG_lmi = numpy.einsum('g,nmg->nmg', SI[ia].conj(), pYlm)
                    #:SPG_lm_aoG = numpy.einsum('nmg,gp->nmp', SPG_lmi, aokG)
                    #:tmp = numpy.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                    #:vppnl += numpy.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
            if p1 > 0:
                SPG_lmi = buf[:p1]
                SPG_lmi *= SI[ia].conj()
                SPG_lm_aoGs = lib.zdot(SPG_lmi, aokG)
                p1 = 0
                for l, proj in enumerate(pp[5:]):
                    rl, nl, hl = proj
                    if nl > 0:
                        p0, p1 = p1, p1+nl*(l*2+1)
                        hl = numpy.asarray(hl)
                        SPG_lm_aoG = SPG_lm_aoGs[p0:p1].reshape(nl,l*2+1,-1)
                        tmp = numpy.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                        vppnl += numpy.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
        return vppnl * (1./ngrids**2)

    vppnl = []
    for kpt in mpi.static_partition(kpts_lst):
        vppnl.append(vppnl_by_k(kpt))
    vppnl = mpi.gather(lib.asarray(vppnl))

    if rank == 0:
        for k in range(nkpts):
            if dtype == numpy.float64:
                vpp[k] += vppnl[k].real
            else:
                vpp[k] += vppnl[k]
        if kpts is None or numpy.shape(kpts) == (3,):
            vpp = vpp[0]
        return vpp

def _sync_mydf(mydf):
    mydf.unpack_(comm.bcast(mydf.pack()))
    return mydf


@mpi.register_class
class FFTDF(fft.FFTDF):
    '''Density expansion on plane waves
    '''
    def __init__(self, cell, kpts=numpy.zeros((1,3))):
        if (cell.dimension < 2 or
            (cell.dimension == 2 and cell.low_dim_ft_type == 'inf_vacuum')):
            raise RuntimeError('MPI-FFTDF module does not support 0D/1D/2D low-dimension '
                               'PBC system')
        fft.FFTDF.__init__(self, cell, kpts)
        # for occ-fft
        self.occ = False

    def pack(self):
        return {'verbose'   : self.verbose,
                'max_memory': self.max_memory,
                'kpts'      : self.kpts,
                'mesh'      : self.mesh}
    def unpack_(self, dfdic):
        self.__dict__.update(dfdic)
        return self

    def mpi_aoR_loop(self, grids=None, kpts=None, deriv=0):
        if grids is None:
            grids = self.grids
            cell = self.cell
        else:
            cell = grids.cell
        if grids.non0tab is None:
            grids.build(with_non0tab=True)

        if kpts is None: kpts = self.kpts
        kpts = numpy.asarray(kpts)

        max_memory = max(2000, self.max_memory-lib.current_memory()[0])
        ni = self._numint
        nao = cell.nao_nr()
        ngrids = grids.weights.size
        nblks = (ngrids+gen_grid.BLKSIZE-1)//gen_grid.BLKSIZE
        mpi_size = mpi.pool.size
        step = (nblks+mpi_size-1) // mpi_size * gen_grid.BLKSIZE
        start = min(ngrids, rank * step)
        stop = min(ngrids, start + step)
        grids = copy.copy(grids)
        grids.coords = grids.coords[start:stop]
        grids.weights = grids.weights[start:stop]
        grids.non0tab = grids.non0tab[start:stop]

        p1 = start
        for ao_k1_etc in ni.block_loop(cell, grids, nao, deriv, kpts,
                                       max_memory=max_memory):
            coords = ao_k1_etc[4]
            p0, p1 = p1, p1 + coords.shape[0]
            yield ao_k1_etc, p0, p1

    get_pp = get_pp
    get_nuc = get_nuc

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, exxdiv='ewald'):
        if kpts is None:
            if numpy.all(self.kpts == 0):
                # Gamma-point calculation by default
                kpts = numpy.zeros(3)
            else:
                kpts = self.kpts
        else:
            kpts = numpy.asarray(kpts)

        vj = vk = None
        if kpts.shape == (3,):
            if with_k:
                vk = mpi_fft_jk.get_k(self, dm, hermi, kpts, kpts_band, exxdiv)
            if with_j:
                vj = mpi_fft_jk.get_j(self, dm, hermi, kpts, kpts_band)
        else:
            if with_k:
                if self.occ == False:
                    vk = mpi_fft_jk.get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv)
                else:
                    vk = mpi_fft_occk.get_k_kpts_occ(self, dm, hermi, kpts, kpts_band, exxdiv)
            if with_j:
                vj = mpi_fft_jk.get_j_kpts(self, dm, hermi, kpts, kpts_band)
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
    cell.h = numpy.eye(3) * 2.5
    cell.mesh = [11] * 3
    cell.build()
    numpy.random.seed(19)
    kpts = numpy.random.random((5,3))

    mydf = df.FFTDF(cell)
    v = mydf.get_nuc()
    print(v.shape)
    v = mydf.get_pp(kpts)
    print(v.shape)

    cell = pgto.M(atom='He 0 0 0; He 0 0 1', h=numpy.eye(3)*4, mesh=[11]*3)
    mydf = df.FFTDF(cell)
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

