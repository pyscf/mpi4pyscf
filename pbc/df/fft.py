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
from mpi4pyscf.pbc.df import fft_jk

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
    if abs(kpts_lst).sum < 1e-9:
        dtype = numpy.float64
    else:
        dtype = numpy.complex128

    gs = mydf.gs
    charge = -cell.atom_charges()
    Gv = cell.get_Gv(gs)
    SI = cell.get_SI(Gv)
    rhoG = numpy.dot(charge, SI)

    coulG = tools.get_coulG(cell, gs=gs, Gv=Gv)
    vneG = rhoG * coulG
    vneR = tools.ifft(vneG, mydf.gs).real

    vne = [lib.dot(aoR.T.conj()*vneR, aoR)
           for k, aoR in mydf.mpi_aoR_loop(cell, gs, kpts_lst)]
    vne = mpi.gather(lib.asarray(vne, dtype=dtype))

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
    if abs(kpts_lst).sum < 1e-9:
        dtype = numpy.float64
    else:
        dtype = numpy.complex128

    gs = mydf.gs
    SI = cell.get_SI()
    Gv = cell.get_Gv(gs)
    vpplocG = pseudo.get_vlocG(cell, Gv)
    vpplocG = -numpy.einsum('ij,ij->j', SI, vpplocG)
    vpplocG[0] = numpy.sum(pseudo.get_alphas(cell)) # from get_jvloc_G0 function
    ngs = len(vpplocG)
    nao = cell.nao_nr()

    # vpploc evaluated in real-space
    vpplocR = tools.ifft(vpplocG, cell.gs).real
    vpp = [lib.dot(aoR.T.conj()*vpplocR, aoR)
           for k, aoR in mydf.mpi_aoR_loop(cell, gs, kpts_lst)]
    vpp = mpi.gather(lib.asarray(vpp, dtype=dtype))

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

    def vppnl_by_k(kpt):
        Gk = Gv + kpt
        G_rad = lib.norm(Gk, axis=1)
        aokG = ft_ao.ft_ao(cell, Gv, kpt=kpt) * (ngs/cell.vol)
        vppnl = 0
        for ia in range(cell.natm):
            symb = cell.atom_symbol(ia)
            if symb not in cell._pseudo:
                continue
            pp = cell._pseudo[symb]
            for l, proj in enumerate(pp[5:]):
                rl, nl, hl = proj
                if nl > 0:
                    hl = numpy.asarray(hl)
                    fakemol._bas[0,gto.ANG_OF] = l
                    fakemol._env[ptr+3] = .5*rl**2
                    fakemol._env[ptr+4] = rl**(l+1.5)*numpy.pi**1.25
                    pYlm_part = dft.numint.eval_ao(fakemol, Gk, deriv=0)

                    pYlm = numpy.empty((nl,l*2+1,ngs))
                    for k in range(nl):
                        qkl = pseudo.pp._qli(G_rad*rl, l, k)
                        pYlm[k] = pYlm_part.T * qkl
                    # pYlm is real
                    SPG_lmi = numpy.einsum('g,nmg->nmg', SI[ia].conj(), pYlm)
                    SPG_lm_aoG = numpy.einsum('nmg,gp->nmp', SPG_lmi, aokG)
                    tmp = numpy.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                    vppnl += numpy.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
        return vppnl * (1./ngs**2)

    vppnl = []
    for kpt in mpi.static_partition(kpts_lst):
        vppnl.append(vppnl_by_k(kpt))
    vppnl = mpi.gather(lib.asarray(vppnl, dtype=dtype))

    if rank == 0:
        vpp += vppnl
        if kpts is None or numpy.shape(kpts) == (3,):
            vpp = vpp[0]
        return vpp

def _sync_mydf(mydf):
    mydf.unpack_(comm.bcast(mydf.pack()))
    return mydf


class _FFTDF(fft.FFTDF):
    '''Density expansion on plane waves
    '''
    def __enter__(self):
        return self
    def __exit__(self):
        self.close()
    def close(self):
        self._reg_keys = mpi.del_registry(self._reg_keys)

    def pack(self):
        return {'verbose'   : self.verbose,
                'max_memory': self.max_memory,
                'kpts'      : self.kpts,
                'gs'        : self.gs,
                'exxdiv'    : self.exxdiv}
    def unpack_(self, dfdic):
        self.__dict__.update(dfdic)
        return self

    def mpi_aoR_loop(self, cell, gs=None, kpts=None, kpt_band=None):
        if kpts is None: kpts = self.kpts
        kpts = numpy.asarray(kpts)
        nkpts = len(kpts)

        if gs is None:
            gs = self.gs
        else:
            self.gs = gs
        ngrids = numpy.prod(numpy.asarray(gs)*2+1)

        if (self._numint.cell is None or id(cell) != id(self._numint.cell) or
            self._numint._deriv != 0 or
            self._numint._kpts.shape != kpts.shape or
            abs(self._numint._kpts - kpts).sum() > 1e-9 or
            self._numint._coords.shape[0] != ngrids or
            (gs is not None and numpy.any(gs != self.gs))):

            coords = gen_grid.gen_uniform_grids(cell, gs)
            nao = cell.nao_nr()
            blksize = int(self.max_memory*1e6/(nkpts*nao*16*numint.BLKSIZE))*numint.BLKSIZE
            blksize = min(max(blksize, numint.BLKSIZE), ngrids)
            try:
                self._numint.cache_ao(cell, kpts, 0, coords, nao, blksize)
            except IOError as e:
                sys.stderr.write('HDF5 file cannot be opened twice in different mode.\n')
                raise e

        with h5py.File(self._numint._ao.name, 'r') as f:
            if kpt_band is None:
                for k in mpi.static_partition(range(nkpts)):
                    aoR = f['ao/%d'%k].value
                    yield k, aoR
            else:
                kpt_band = numpy.reshape(kpt_band, 3)
                where = numpy.argmin(pyscf.lib.norm(kpts-kpt_band,axis=1))
                if abs(kpts[where]-kpt_band).sum() > 1e-9:
                    where = None
                    coords = gen_grid.gen_uniform_grids(cell, gs)
                    yield 0, numint.eval_ao(cell, coords, kpt_band, deriv=deriv)
                else:
                    yield where, f['ao/%d'%where].value

    get_nuc = get_nuc
    get_pp = get_pp

    def get_jk(self, dm, hermi=1, kpts=None, kpt_band=None,
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
                vk = fft_jk.get_k(self, dm, hermi, kpts, kpt_band, exxdiv)
            if with_j:
                vj = fft_jk.get_j(self, dm, hermi, kpts, kpt_band)
        else:
            if with_k:
                vk = fft_jk.get_k_kpts(self, dm, hermi, kpts, kpt_band, exxdiv)
            if with_j:
                vj = fft_jk.get_j_kpts(self, dm, hermi, kpts, kpt_band)
        return vj, vk

FFTDF = mpi.register_class(_FFTDF)


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

    mydf = df.FFTDF(cell)
    v = mydf.get_nuc()
    print(v.shape)
    v = mydf.get_pp(kpts)
    print(v.shape)

    cell = pgto.M(atom='He 0 0 0; He 0 0 1', h=numpy.eye(3)*4, gs=[5]*3)
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

