#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Mixed density fitting with Gaussian and planewaves
Ref:
'''

import os
import time
import tempfile
import ctypes
import numpy
import scipy.linalg
import h5py

from pyscf import lib
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import mdf
from pyscf.pbc.df.incore import wrap_int3c
from pyscf.pbc.df.df import fuse_auxcell, unique
from pyscf.pbc.df.df_jk import zdotCN, is_zero, gamma_point
from pyscf.pbc.df.aft import _sub_df_jk_
from pyscf.gto.mole import PTR_COORD
from pyscf.ao2mo.outcore import balance_segs

from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi
from mpi4pyscf.pbc.df import mdf_jk as mpi_mdf_jk
from mpi4pyscf.pbc.df import mdf_ao2mo as mpi_mdf_ao2mo
from mpi4pyscf.pbc.df import df as mpi_df
from mpi4pyscf.pbc.df import aft as mpi_aft

comm = mpi.comm
rank = mpi.rank


def _make_j3c(mydf, cell, auxcell, kptij_lst, cderi_file):
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = t0 = (time.clock(), time.time())

    fused_cell, fuse = fuse_auxcell(mydf, mydf.auxcell)
    ao_loc = cell.ao_loc_nr()
    nao = ao_loc[-1]
    naux = auxcell.nao_nr()
    nkptij = len(kptij_lst)
    mesh = mydf.mesh
    Gv, Gvbase, kws = cell.get_Gv_weights(mesh)
    b = cell.reciprocal_vectors()
    gxyz = lib.cartesian_prod([numpy.arange(len(x)) for x in Gvbase])
    ngrids = gxyz.shape[0]

    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    kpt_ji = kptjs - kptis
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)
    log.debug('Num uniq kpts %d', len(uniq_kpts))
    log.debug2('uniq_kpts %s', uniq_kpts)
    # j2c ~ (-kpt_ji | kpt_ji)
    j2c = fused_cell.pbc_intor('int2c2e', hermi=1, kpts=uniq_kpts)
    j2ctags = []
    t1 = log.timer_debug1('2c2e', *t1)

    swapfile = tempfile.NamedTemporaryFile(dir=os.path.dirname(cderi_file))
    fswap = lib.H5TmpFile(swapfile.name)
    # Unlink swapfile to avoid trash
    swapfile = None

    for k, kpt in enumerate(uniq_kpts):
        coulG = mydf.weighted_coulG(kpt, False, mesh)
        j2c[k] = fuse(fuse(j2c[k]).T).T.copy()
        j2c_k = numpy.zeros_like(j2c[k])
        for p0, p1 in mydf.mpi_prange(0, ngrids):
            aoaux = ft_ao.ft_ao(fused_cell, Gv[p0:p1], None, b, gxyz[p0:p1], Gvbase, kpt).T
            aoaux = fuse(aoaux)
            LkR = numpy.asarray(aoaux.real, order='C')
            LkI = numpy.asarray(aoaux.imag, order='C')
            aoaux = None

            if is_zero(kpt):  # kpti == kptj
                j2cR   = lib.dot(LkR*coulG[p0:p1], LkR.T)
                j2c_k += lib.dot(LkI*coulG[p0:p1], LkI.T, 1, j2cR, 1)
            else:
                # aoaux ~ kpt_ij, aoaux.conj() ~ kpt_kl
                j2cR, j2cI = zdotCN(LkR*coulG[p0:p1], LkI*coulG[p0:p1], LkR.T, LkI.T)
                j2c_k += j2cR + j2cI * 1j
            LkR = LkI = None
        j2c[k] -= mpi.allreduce(j2c_k)

        try:
            fswap['j2c/%d'%k] = scipy.linalg.cholesky(j2c[k], lower=True)
            j2ctags.append('CD')
        except scipy.linalg.LinAlgError:
            w, v = scipy.linalg.eigh(j2c[k])
            log.debug2('metric linear dependency for kpt %s', k)
            log.debug2('cond = %.4g, drop %d bfns',
                       w[0]/w[-1], numpy.count_nonzero(w<mydf.linear_dep_threshold))
            v1 = v[:,w>mydf.linear_dep_threshold].T.conj()
            v1 /= numpy.sqrt(w[w>mydf.linear_dep_threshold]).reshape(-1,1)
            fswap['j2c/%d'%k] = v1
            if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
                idx = numpy.where(w < -mydf.linear_dep_threshold)[0]
                if len(idx) > 0:
                    fswap['j2c-/%d'%k] = (v[:,idx]/numpy.sqrt(-w[idx])).conj().T
            w = v = v1 = v2 = None
            j2ctags.append('eig')
        aoaux = kLR = kLI = j2cR = j2cI = coulG = None
    j2c = None

    aosym_s2 = numpy.einsum('ix->i', abs(kptis-kptjs)) < 1e-9
    j_only = numpy.all(aosym_s2)
    if gamma_point(kptij_lst):
        dtype = 'f8'
    else:
        dtype = 'c16'
    t1 = log.timer_debug1('aoaux and int2c', *t1)

# Estimates the buffer size based on the last contraction in G-space.
# This contraction requires to hold nkptj copies of (naux,?) array
# simultaneously in memory.
    mem_now = max(comm.allgather(lib.current_memory()[0]))
    max_memory = max(2000, mydf.max_memory - mem_now)
    nkptj_max = max((uniq_inverse==x).sum() for x in set(uniq_inverse))
    buflen = max(int(min(max_memory*.5e6/16/naux/(nkptj_max+2)/nao,
                         nao/3/mpi.pool.size)), 1)
    chunks = (buflen, nao)

    j3c_jobs = mpi_df.grids2d_int3c_jobs(cell, auxcell, kptij_lst, chunks, j_only)
    log.debug1('max_memory = %d MB (%d in use)  chunks %s',
               max_memory, mem_now, chunks)
    log.debug2('j3c_jobs %s', j3c_jobs)

    if j_only:
        int3c = wrap_int3c(cell, fused_cell, 'int3c2e', 's2', 1, kptij_lst)
    else:
        int3c = wrap_int3c(cell, fused_cell, 'int3c2e', 's1', 1, kptij_lst)
        idxb = numpy.tril_indices(nao)
        idxb = (idxb[0] * nao + idxb[1]).astype('i')
    aux_loc = fused_cell.ao_loc_nr(fused_cell.cart)

    def gen_int3c(job_id, ish0, ish1):
        dataname = 'j3c-chunks/%d' % job_id
        i0 = ao_loc[ish0]
        i1 = ao_loc[ish1]
        dii = i1*(i1+1)//2 - i0*(i0+1)//2
        dij = (i1 - i0) * nao
        if j_only:
            buflen = max(8, int(max_memory*1e6/16/(nkptij*dii+dii)))
        else:
            buflen = max(8, int(max_memory*1e6/16/(nkptij*dij+dij)))
        auxranges = balance_segs(aux_loc[1:]-aux_loc[:-1], buflen)
        buflen = max([x[2] for x in auxranges])
        buf = numpy.empty(nkptij*dij*buflen, dtype=dtype)
        buf1 = numpy.empty(dij*buflen, dtype=dtype)

        naux = aux_loc[-1]
        for kpt_id, kptij in enumerate(kptij_lst):
            key = '%s/%d' % (dataname, kpt_id)
            if aosym_s2[kpt_id]:
                shape = (naux, dii)
            else:
                shape = (naux, dij)
            if gamma_point(kptij):
                fswap.create_dataset(key, shape, 'f8')
            else:
                fswap.create_dataset(key, shape, 'c16')

        naux0 = 0
        for istep, auxrange in enumerate(auxranges):
            log.alldebug2("aux_e1 job_id %d step %d", job_id, istep)
            sh0, sh1, nrow = auxrange
            sub_slice = (ish0, ish1, 0, cell.nbas, sh0, sh1)
            if j_only:
                mat = numpy.ndarray((nkptij,dii,nrow), dtype=dtype, buffer=buf)
            else:
                mat = numpy.ndarray((nkptij,dij,nrow), dtype=dtype, buffer=buf)
            mat = int3c(sub_slice, mat)

            for k, kptij in enumerate(kptij_lst):
                h5dat = fswap['%s/%d'%(dataname,k)]
                v = lib.transpose(mat[k], out=buf1)
                if not j_only and aosym_s2[k]:
                    idy = idxb[i0*(i0+1)//2:i1*(i1+1)//2] - i0 * nao
                    out = numpy.ndarray((nrow,dii), dtype=v.dtype, buffer=mat[k])
                    v = numpy.take(v, idy, axis=1, out=out)
                if gamma_point(kptij):
                    h5dat[naux0:naux0+nrow] = v.real
                else:
                    h5dat[naux0:naux0+nrow] = v
            naux0 += nrow

    def ft_fuse(job_id, uniq_kptji_id, sh0, sh1):
        kpt = uniq_kpts[uniq_kptji_id]  # kpt = kptj - kpti
        adapted_ji_idx = numpy.where(uniq_inverse == uniq_kptji_id)[0]
        adapted_kptjs = kptjs[adapted_ji_idx]
        nkptj = len(adapted_kptjs)

        Gaux = ft_ao.ft_ao(fused_cell, Gv, None, b, gxyz, Gvbase, kpt).T
        Gaux = fuse(Gaux)
        Gaux *= mydf.weighted_coulG(kpt, False, mesh)
        kLR = lib.transpose(numpy.asarray(Gaux.real, order='C'))
        kLI = lib.transpose(numpy.asarray(Gaux.imag, order='C'))
        j2c = numpy.asarray(fswap['j2c/%d'%uniq_kptji_id])
        j2ctag = j2ctags[uniq_kptji_id]
        naux0 = j2c.shape[0]
        if ('j2c-/%d' % uniq_kptji_id) in fswap:
            j2c_negative = numpy.asarray(fswap['j2c-/%d'%uniq_kptji_id])
        else:
            j2c_negative = None

        if is_zero(kpt):
            aosym = 's2'
        else:
            aosym = 's1'

        if aosym == 's2' and cell.dimension == 3:
            vbar = fuse(mydf.auxbar(fused_cell))
            ovlp = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=adapted_kptjs)
            ovlp = [lib.pack_tril(s) for s in ovlp]

        j3cR = [None] * nkptj
        j3cI = [None] * nkptj
        i0 = ao_loc[sh0]
        i1 = ao_loc[sh1]
        for k, idx in enumerate(adapted_ji_idx):
            key = 'j3c-chunks/%d/%d' % (job_id, idx)
            v = fuse(numpy.asarray(fswap[key]))
            if aosym == 's2' and cell.dimension == 3:
                for i in numpy.where(vbar != 0)[0]:
                    v[i] -= vbar[i] * ovlp[k][i0*(i0+1)//2:i1*(i1+1)//2].ravel()
            j3cR[k] = numpy.asarray(v.real, order='C')
            if v.dtype == numpy.complex128:
                j3cI[k] = numpy.asarray(v.imag, order='C')
            v = None

        ncol = j3cR[0].shape[1]
        Gblksize = max(16, int(max_memory*1e6/16/ncol/(nkptj+1)))  # +1 for pqkRbuf/pqkIbuf
        Gblksize = min(Gblksize, ngrids, 16384)
        pqkRbuf = numpy.empty(ncol*Gblksize)
        pqkIbuf = numpy.empty(ncol*Gblksize)
        buf = numpy.empty(nkptj*ncol*Gblksize, dtype=numpy.complex128)
        log.alldebug2('    blksize (%d,%d)', Gblksize, ncol)

        if aosym == 's2':
            shls_slice = (sh0, sh1, 0, sh1)
        else:
            shls_slice = (sh0, sh1, 0, cell.nbas)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            dat = ft_ao._ft_aopair_kpts(cell, Gv[p0:p1], shls_slice, aosym, b,
                                        gxyz[p0:p1], Gvbase, kpt,
                                        adapted_kptjs, out=buf)
            nG = p1 - p0
            for k, ji in enumerate(adapted_ji_idx):
                aoao = dat[k].reshape(nG,ncol)
                pqkR = numpy.ndarray((ncol,nG), buffer=pqkRbuf)
                pqkI = numpy.ndarray((ncol,nG), buffer=pqkIbuf)
                pqkR[:] = aoao.real.T
                pqkI[:] = aoao.imag.T

                lib.dot(kLR[p0:p1].T, pqkR.T, -1, j3cR[k], 1)
                lib.dot(kLI[p0:p1].T, pqkI.T, -1, j3cR[k], 1)
                if not (is_zero(kpt) and gamma_point(adapted_kptjs[k])):
                    lib.dot(kLR[p0:p1].T, pqkI.T, -1, j3cI[k], 1)
                    lib.dot(kLI[p0:p1].T, pqkR.T,  1, j3cI[k], 1)

        for k, idx in enumerate(adapted_ji_idx):
            if is_zero(kpt) and gamma_point(adapted_kptjs[k]):
                v = j3cR[k]
            else:
                v = j3cR[k] + j3cI[k] * 1j
            if j2ctag == 'CD':
                v = scipy.linalg.solve_triangular(j2c, v, lower=True, overwrite_b=True)
                fswap['j3c-chunks/%d/%d'%(job_id,idx)][:naux0] = v
            else:
                fswap['j3c-chunks/%d/%d'%(job_id,idx)][:naux0] = lib.dot(j2c, v)

            # low-dimension systems
            if j2c_negative is not None:
                fswap['j3c-/%d/%d'%(job_id,idx)] = lib.dot(j2c_negative, v)

    mpi_df._assemble(mydf, kptij_lst, j3c_jobs, gen_int3c, ft_fuse, cderi_file, fswap, log)


@mpi.register_class
class MDF(mdf.MDF, mpi_df.DF):

    def pack(self):
        return {'verbose'   : self.verbose,
                'max_memory': self.max_memory,
                'kpts'      : self.kpts,
                'kpts_band' : self.kpts_band,
                'mesh'      : self.mesh,
                'blockdim'  : self.blockdim,
                '_auxbasis' : self._auxbasis,
                '_eta'      : self._eta,
                '_exp_to_discard' : self._exp_to_discard,
                'linear_dep_threshold': self.linear_dep_threshold,
                '_cderi'     : self._cderi}

    _make_j3c = _make_j3c

    get_nuc = mpi_aft.get_nuc
    get_pp = mpi_aft.get_pp
    _int_nuc_vloc = mpi_aft._int_nuc_vloc

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv='ewald'):
        # J/K for RSH functionals
        if omega is not None:
            return _sub_df_jk_(self, dm, hermi, kpts, kpts_band,
                               with_j, with_k, omega, exxdiv)

        if kpts is None:
            if numpy.all(self.kpts == 0):
                # Gamma-point calculation by default
                kpts = numpy.zeros(3)
            else:
                kpts = self.kpts
        else:
            kpts = numpy.asarray(kpts)

        if kpts.shape == (3,):
            return mpi_mdf_jk.get_jk(self, dm, hermi, kpts, kpts_band, with_j,
                                     with_k, exxdiv)

        vj = vk = None
        if with_k:
            vk = mpi_mdf_jk.get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv)
        if with_j:
            vj = mpi_mdf_jk.get_j_kpts(self, dm, hermi, kpts, kpts_band)
        return vj, vk

    get_eri = get_ao_eri = mpi_mdf_ao2mo.get_eri
    ao2mo = get_mo_eri = mpi_mdf_ao2mo.general


    def loop(self):
        # mpi.pool.worker_status = P (pending) means the caller on master
        # process runs in serial mode.
        # 3-index tensor should be generated on each process and sent to
        # the master process.  E.g. the call to with_df.loop in dfccsd
        serial_mode = mpi.pool.worker_status == 'P'
        if serial_mode:
            return loop_yield_then_reduce(self)
        else:
            return mdf.MDF.loop(self)

    def get_naoaux(self):
        if mpi.pool.worker_status == 'P':
            return mpi_df.get_naoaux(self) + mpi_aft.AFTDF.get_naoaux(self)
        else:
            return mpi_df.DF.get_naoaux(self)


def _sync_mydf(mydf):
    return mydf.unpack_(comm.bcast(mydf.pack()))

@mpi.reduced_yield
def loop_yield_then_reduce(mydf):
    for Lpq in mdf.MDF.loop(mydf):
        yield Lpq

if __name__ == '__main__':
    from pyscf.pbc import gto as pgto
    from mpi4pyscf.pbc import df
    cell = pgto.M(atom='He 0 0 0; He 0 0 1', a=numpy.eye(3)*4, mesh=[11]*3)
    mydf = df.MDF(cell, kpts)

    v = mydf.get_nuc()
    print(v.shape)
    #v = mydf.get_pp(kpts)
    #print(v.shape)

    nao = cell.nao_nr()
    dm = numpy.ones((nao,nao))
    vj, vk = mydf.get_jk(dm, kpts=kpts[0])
    print(vj.shape)
    print(vk.shape)

    dm_kpts = [dm]*5
    vj, vk = mydf.get_jk(dm_kpts, kpts=kpts)
    print(vj.shape)
    print(vk.shape)

    mydf.close()

