#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import platform
import ctypes
import numpy
import scipy.linalg
import h5py

from pyscf import lib
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import df
from pyscf.pbc.df.incore import wrap_int3c
from pyscf.pbc.df.df import fuse_auxcell, make_modrho_basis, unique
from pyscf.pbc.df.df_jk import zdotCN, is_zero, gamma_point
from pyscf.gto.mole import PTR_COORD
from pyscf.ao2mo.outcore import balance_segs

from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi
from mpi4pyscf.pbc.df import df_jk
from mpi4pyscf.pbc.df import df_ao2mo
from mpi4pyscf.pbc.df import aft

comm = mpi.comm
rank = mpi.rank


@mpi.parallel_call
def build(mydf, j_only=None, with_j3c=True, kpts_band=None):
# Unlike DF and AFT class, here MDF objects are synced once
    if mpi.pool.size == 1:
        return df.DF.build(mydf, j_only, with_j3c, kpts_band)

    mydf = _sync_mydf(mydf)
    cell = mydf.cell
    log = logger.Logger(mydf.stdout, mydf.verbose)
    info = rank, platform.node(), platform.os.getpid()
    log.debug('MPI info (rank, host, pid)  %s', comm.gather(info))

    t1 = (time.clock(), time.time())
    if mydf.kpts_band is not None:
        mydf.kpts_band = numpy.reshape(mydf.kpts_band, (-1,3))
    if kpts_band is not None:
        kpts_band = numpy.reshape(kpts_band, (-1,3))
        if mydf.kpts_band is None:
            mydf.kpts_band = kpts_band
        else:
            mydf.kpts_band = unique(numpy.vstack((mydf.kpts_band,kpts_band)))[0]

    mydf.dump_flags()

    mydf.auxcell = make_modrho_basis(cell, mydf.auxbasis, mydf.eta)

    if mydf.kpts_band is None:
        kpts = mydf.kpts
        kband_uniq = numpy.zeros((0,3))
    else:
        kpts = mydf.kpts
        kband_uniq = [k for k in mydf.kpts_band if len(member(k, kpts))==0]
    if j_only is None:
        j_only = mydf._j_only
    if j_only:
        kall = numpy.vstack([kpts,kband_uniq])
        kptij_lst = numpy.hstack((kall,kall)).reshape(-1,2,3)
    else:
        kptij_lst = [(ki, kpts[j]) for i, ki in enumerate(kpts) for j in range(i+1)]
        kptij_lst.extend([(ki, kj) for ki in kband_uniq for kj in kpts])
        kptij_lst.extend([(ki, ki) for ki in kband_uniq])
        kptij_lst = numpy.asarray(kptij_lst)

    if with_j3c:
        if isinstance(self._cderi_to_save, str):
            cderi = self._cderi_to_save
        else:
            cderi = self._cderi_to_save.name
        if isinstance(self._cderi, str):
            log.warn('Value of _cderi is ignored. DF integrals will be '
                     'saved in file %s .', cderi)
        self._cderi = cderi
        mydf._make_j3c(cell, mydf.auxcell, kptij_lst, cderi)
        t1 = log.timer_debug1('j3c', *t1)
    return mydf


def _make_j3c(mydf, cell, auxcell, kptij_lst, cderi_file):
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = t0 = (time.clock(), time.time())

    fused_cell, fuse = fuse_auxcell(mydf, mydf.auxcell)
    ao_loc = cell.ao_loc_nr()
    nao = ao_loc[-1]
    naux = auxcell.nao_nr()
    nkptij = len(kptij_lst)
    gs = mydf.gs
    Gv, Gvbase, kws = cell.get_Gv_weights(gs)
    b = cell.reciprocal_vectors()
    gxyz = lib.cartesian_prod([numpy.arange(len(x)) for x in Gvbase])
    ngs = gxyz.shape[0]

    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    kpt_ji = kptjs - kptis
    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)
    log.debug('Num uniq kpts %d', len(uniq_kpts))
    log.debug2('uniq_kpts %s', uniq_kpts)
    # j2c ~ (-kpt_ji | kpt_ji)
    j2c = fused_cell.pbc_intor('cint2c2e_sph', hermi=1, kpts=uniq_kpts)
    j2ctags = []
    nauxs = []
    t1 = log.timer_debug1('2c2e', *t1)

    if h5py.is_hdf5(cderi_file):
        feri = h5py.File(cderi_file)
    else:
        feri = h5py.File(cderi_file, 'w')
    for k, kpt in enumerate(uniq_kpts):
        aoaux = ft_ao.ft_ao(fused_cell, Gv, None, b, gxyz, Gvbase, kpt).T
        coulG = numpy.sqrt(mydf.weighted_coulG(kpt, False, gs))
        kLR = (aoaux.real * coulG).T
        kLI = (aoaux.imag * coulG).T
        if not kLR.flags.c_contiguous: kLR = lib.transpose(kLR.T)
        if not kLI.flags.c_contiguous: kLI = lib.transpose(kLI.T)
        aoaux = None

        kLR1 = numpy.asarray(kLR[:,naux:], order='C')
        kLI1 = numpy.asarray(kLI[:,naux:], order='C')
        if is_zero(kpt):  # kpti == kptj
            for p0, p1 in mydf.mpi_prange(0, ngs):
                j2cR = lib.ddot(kLR1[p0:p1].T, kLR[p0:p1])
                j2cR = lib.ddot(kLI1[p0:p1].T, kLI[p0:p1], 1, j2cR, 1)
                j2c[k][naux:] -= mpi.allreduce(j2cR)
                j2c[k][:naux,naux:] = j2c[k][naux:,:naux].T
        else:
            for p0, p1 in mydf.mpi_prange(0, ngs):
                j2cR, j2cI = zdotCN(kLR1[p0:p1].T, kLI1[p0:p1].T, kLR[p0:p1], kLI[p0:p1])
                j2cR = mpi.allreduce(j2cR)
                j2cI = mpi.allreduce(j2cI)
                j2c[k][naux:] -= j2cR + j2cI * 1j
                j2c[k][:naux,naux:] = j2c[k][naux:,:naux].T.conj()
        j2c[k] = fuse(fuse(j2c[k]).T).T
        try:
            feri['j2c/%d'%k] = scipy.linalg.cholesky(j2c[k], lower=True)
            j2ctags.append('CD')
            nauxs.append(naux)
        except scipy.linalg.LinAlgError as e:
            #msg =('===================================\n'
            #      'J-metric not positive definite.\n'
            #      'It is likely that gs is not enough.\n'
            #      '===================================')
            #log.error(msg)
            #raise scipy.linalg.LinAlgError('\n'.join([e.message, msg]))
            w, v = scipy.linalg.eigh(j2c)
            log.debug2('metric linear dependency for kpt %s', uniq_kptji_id)
            log.debug2('cond = %.4g, drop %d bfns',
                       w[0]/w[-1], numpy.count_nonzero(w<LINEAR_DEP_THR))
            v = v[:,w>LINEAR_DEP_THR].T.conj()
            v /= numpy.sqrt(w[w>LINEAR_DEP_THR]).reshape(-1,1)
            feri['j2c/%d'%k] = v
            j2ctags.append('eig')
            nauxs.append(v.shape[0])
        kLR = kLI = kLR1 = kLI1 = coulG = None
    j2c = None

    aosym_s2 = numpy.einsum('ix->i', abs(kptis-kptjs)) < 1e-9
    j_only = numpy.all(aosym_s2)
    if gamma_point(kptij_lst):
        dtype = 'f8'
    else:
        dtype = 'c16'
    vbar = mydf.auxbar(fused_cell)
    vbar = fuse(vbar)
    ovlp = cell.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=kptjs[aosym_s2])
    ovlp = [lib.pack_tril(s) for s in ovlp]
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

    j3c_jobs = grids2d_int3c_jobs(cell, auxcell, kptij_lst, chunks, j_only)
    log.debug1('max_memory = %d MB (%d in use)  chunks %s',
               max_memory, mem_now, chunks)
    log.debug2('j3c_jobs %s', j3c_jobs)

    if j_only:
        int3c = wrap_int3c(cell, fused_cell, 'cint3c2e_sph', 's2', 1, kptij_lst)
    else:
        int3c = wrap_int3c(cell, fused_cell, 'cint3c2e_sph', 's1', 1, kptij_lst)
        idxb = numpy.tril_indices(nao)
        idxb = (idxb[0] * nao + idxb[1]).astype('i')
    aux_loc = fused_cell.ao_loc_nr('ssc' in 'cint3c2e_sph')

    def gen_int3c(auxcell, job_id, ish0, ish1):
        dataname = 'j3c-chunks/%d' % job_id
        if dataname in feri:
            del(feri[dataname])

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
                feri.create_dataset(key, shape, 'f8')
            else:
                feri.create_dataset(key, shape, 'c16')

        naux0 = 0
        for istep, auxrange in enumerate(auxranges):
            log.alldebug2("aux_e2 job_id %d step %d", job_id, istep)
            sh0, sh1, nrow = auxrange
            sub_slice = (ish0, ish1, 0, cell.nbas, sh0, sh1)
            if j_only:
                mat = numpy.ndarray((nkptij,dii,nrow), dtype=dtype, buffer=buf)
            else:
                mat = numpy.ndarray((nkptij,dij,nrow), dtype=dtype, buffer=buf)
            mat = int3c(sub_slice, mat)

            for k, kptij in enumerate(kptij_lst):
                h5dat = feri['%s/%d'%(dataname,k)]
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

        shls_slice = (auxcell.nbas, fused_cell.nbas)
        Gaux = ft_ao.ft_ao(fused_cell, Gv, shls_slice, b, gxyz, Gvbase, kpt)
        Gaux *= mydf.weighted_coulG(kpt, False, gs).reshape(-1,1)
        kLR = Gaux.real.copy('C')
        kLI = Gaux.imag.copy('C')
        j2c = numpy.asarray(feri['j2c/%d'%uniq_kptji_id])
        j2ctag = j2ctags[uniq_kptji_id]
        naux0 = j2c.shape[0]

        if is_zero(kpt):
            aosym = 's2'
        else:
            aosym = 's1'

        j3cR = [None] * nkptj
        j3cI = [None] * nkptj
        i0 = ao_loc[sh0]
        i1 = ao_loc[sh1]
        for k, idx in enumerate(adapted_ji_idx):
            key = 'j3c-chunks/%d/%d' % (job_id, idx)
            v = numpy.asarray(feri[key])
            if is_zero(kpt):
                for i, c in enumerate(vbar):
                    if c != 0:
                        v[i] -= c * ovlp[k][i0*(i0+1)//2:i1*(i1+1)//2].ravel()
            j3cR[k] = numpy.asarray(v.real, order='C')
            if v.dtype == numpy.complex128:
                j3cI[k] = numpy.asarray(v.imag, order='C')
            v = None

        ncol = j3cR[0].shape[1]
        Gblksize = max(16, int(max_memory*1e6/16/ncol/(nkptj+1)))  # +1 for pqkRbuf/pqkIbuf
        Gblksize = min(Gblksize, ngs, 16384)
        pqkRbuf = numpy.empty(ncol*Gblksize)
        pqkIbuf = numpy.empty(ncol*Gblksize)
        buf = numpy.empty(nkptj*ncol*Gblksize, dtype=numpy.complex128)
        log.alldebug2('    blksize (%d,%d)', Gblksize, ncol)

        shls_slice = (sh0, sh1, 0, cell.nbas)
        for p0, p1 in lib.prange(0, ngs, Gblksize):
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

                lib.dot(kLR[p0:p1].T, pqkR.T, -1, j3cR[k][naux:], 1)
                lib.dot(kLI[p0:p1].T, pqkI.T, -1, j3cR[k][naux:], 1)
                if not (is_zero(kpt) and gamma_point(adapted_kptjs[k])):
                    lib.dot(kLR[p0:p1].T, pqkI.T, -1, j3cI[k][naux:], 1)
                    lib.dot(kLI[p0:p1].T, pqkR.T,  1, j3cI[k][naux:], 1)

        for k, idx in enumerate(adapted_ji_idx):
            if is_zero(kpt) and gamma_point(adapted_kptjs[k]):
                v = fuse(j3cR[k])
            else:
                v = fuse(j3cR[k] + j3cI[k] * 1j)
            if j2ctag == 'CD':
                v = scipy.linalg.solve_triangular(j2c, v, lower=True, overwrite_b=True)
            else:
                v = lib.dot(j2c, v)
            feri['j3c-chunks/%d/%d'%(job_id,idx)][:naux0] = v

    t2 = t1
    j3c_workers = numpy.zeros(len(j3c_jobs), dtype=int)
    #for job_id, ish0, ish1 in mpi.work_share_partition(j3c_jobs):
    for job_id, ish0, ish1 in mpi.work_stealing_partition(j3c_jobs):
        gen_int3c(fused_cell, job_id, ish0, ish1)
        t2 = log.alltimer_debug2('int j3c %d' % job_id, *t2)

        for k, kpt in enumerate(uniq_kpts):
            ft_fuse(job_id, k, ish0, ish1)
            t2 = log.alltimer_debug2('ft-fuse %d k %d' % (job_id, k), *t2)

        j3c_workers[job_id] = rank
    j3c_workers = mpi.allreduce(j3c_workers)
    log.debug2('j3c_workers %s', j3c_workers)
    j2c = kLRs = kLIs = ovlp = vbar = fuse = gen_int3c = ft_fuse = None
    t1 = log.timer_debug1('int3c and fuse', *t1)

    if 'j3c' in feri: del(feri['j3c'])
    segsize = (max(nauxs)+mpi.pool.size-1) // mpi.pool.size
    naux0 = rank * segsize
    for k, kptij in enumerate(kptij_lst):
        naux1 = min(nauxs[uniq_inverse[k]], naux0+segsize)
        nrow = max(0, naux1-naux0)
        if gamma_point(kptij):
            dtype = 'f8'
        else:
            dtype = 'c16'
        if aosym_s2[k]:
            nao_pair = nao * (nao+1) // 2
        else:
            nao_pair = nao * nao
        feri.create_dataset('j3c/%d'%k, (nrow,nao_pair), dtype, maxshape=(None,nao_pair))

    off0 = numpy.asarray([ao_loc[i0] for x,i0,i1 in j3c_jobs])
    off1 = numpy.asarray([ao_loc[i1] for x,i0,i1 in j3c_jobs])
    if j_only:
        dims = off1*(off1+1)//2 - off0*(off0+1)//2
    else:
        dims = (off1-off0) * nao
    #dims = numpy.asarray([ao_loc[i1]-ao_loc[i0] for x,i0,i1 in j3c_jobs])
    dims = numpy.hstack([dims[j3c_workers==w] for w in range(mpi.pool.size)])
    job_idx = numpy.hstack([numpy.where(j3c_workers==w)[0]
                            for w in range(mpi.pool.size)])
    segs_loc = numpy.append(0, numpy.cumsum(dims))
    segs_loc = [(segs_loc[j], segs_loc[j+1]) for j in numpy.argsort(job_idx)]
    def load(k, p0, p1):
        naux1 = nauxs[uniq_inverse[k]]
        slices = [(min(i*segsize+p0,naux1), min(i*segsize+p1,naux1))
                  for i in range(mpi.pool.size)]
        segs = []
        for p0, p1 in slices:
            val = []
            for job_id, worker in enumerate(j3c_workers):
                if rank == worker:
                    key = 'j3c-chunks/%d/%d' % (job_id, k)
                    val.append(feri[key][p0:p1].ravel())
            if val:
                segs.append(numpy.hstack(val))
            else:
                segs.append(numpy.zeros(0))
        return segs

    def save(k, p0, p1, segs):
        segs = mpi.alltoall(segs)
        naux1 = nauxs[uniq_inverse[k]]
        loc0, loc1 = min(p0, naux1-naux0), min(p1, naux1-naux0)
        nL = loc1 - loc0
        if nL > 0:
            segs = [segs[i0*nL:i1*nL].reshape(nL,-1) for i0,i1 in segs_loc]
            segs = numpy.hstack(segs)
            feri['j3c/%d'%k][loc0:loc1] = segs

    mem_now = max(comm.allgather(lib.current_memory()[0]))
    max_memory = max(2000, min(8000, mydf.max_memory - mem_now))
    if numpy.all(aosym_s2):
        if gamma_point(kptij_lst):
            blksize = max(16, int(max_memory*.5e6/8/nao**2))
        else:
            blksize = max(16, int(max_memory*.5e6/16/nao**2))
    else:
        blksize = max(16, int(max_memory*.5e6/16/nao**2/2))
    log.debug1('max_momory %d MB (%d in use), blksize %d',
               max_memory, mem_now, blksize)

    t2 = t1
    write_handler = None
    for k, kptji in enumerate(kptij_lst):
        for p0, p1 in lib.prange(0, segsize, blksize):
            segs = load(k, p0, p1)
            write_handler = async_write(write_handler, save, k, p0, p1, segs)
            segs = None
            t2 = log.timer_debug1('assemble k=%d %d:%d (in %d)' %
                                  (k, p0, p1, segsize), *t2)
    if write_handler is not None:
        write_handler.join()
    write_handler = None

    if 'j3c-chunks' in feri: del(feri['j3c-chunks'])
    if 'j3c-kptij' in feri: del(feri['j3c-kptij'])
    feri['j3c-kptij'] = kptij_lst
    t1 = log.alltimer_debug1('assembling j3c', *t1)
    feri.close()


@mpi.register_class
class DF(df.DF, aft.AFTDF):

    build = build
    _make_j3c = _make_j3c

    get_nuc = aft.get_nuc
    _int_nuc_vloc = aft._int_nuc_vloc

    def dump_flags(self):
        return df.DF.dump_flags(logger.Logger(self.stdout, self.verbose))

    def pack(self):
        return {'verbose'   : self.verbose,
                'max_memory': self.max_memory,
                'kpts'      : self.kpts,
                'kpts_band' : self.kpts_band,
                'gs'        : self.gs,
                'eta'       : self.eta,
                'blockdim'  : self.blockdim,
                'auxbasis'  : self.auxbasis}
    def unpack_(self, dfdic):
        self.__dict__.update(dfdic)
        return self

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

        if kpts.shape == (3,):
            return df_jk.get_jk(self, dm, hermi, kpts, kpts_band, with_j,
                                with_k, exxdiv)

        vj = vk = None
        if with_k:
            vk = df_jk.get_k_kpts(self, dm, hermi, kpts, kpts_band, exxdiv)
        if with_j:
            vj = df_jk.get_j_kpts(self, dm, hermi, kpts, kpts_band)
        return vj, vk

    get_eri = get_ao_eri = df_ao2mo.get_eri
    ao2mo = get_mo_eri = df_ao2mo.general

def grids2d_int3c_jobs(cell, auxcell, kptij_lst, chunks, aosym_s2):
    ao_loc = cell.ao_loc_nr()
    if aosym_s2:
        segs = ao_loc[1:]*(ao_loc[1:]+1)//2 - ao_loc[:-1]*(ao_loc[:-1]+1)//2
        ij_ranges = balance_segs(segs, chunks[0]*chunks[1])
    else:
        segs = ao_loc[1:]-ao_loc[:-1]
        ij_ranges = balance_segs(segs, chunks[0])

    jobs = [(job_id, i0, i1) for job_id, (i0, i1, x) in enumerate(ij_ranges)]
    return jobs

def _sync_mydf(mydf):
    return mydf.unpack_(comm.bcast(mydf.pack()))

def async_write(thread_io, fn, *args):
    if thread_io is not None:
        thread_io.join()
    thread_io = lib.background_thread(fn, *args)
    return thread_io


if __name__ == '__main__':
    from pyscf.pbc import gto as pgto
    from mpi4pyscf.pbc import df
    cell = pgto.M(atom='He 0 0 0; He 0 0 1', a=numpy.eye(3)*4, gs=[5]*3)
    mydf = df.DF(cell, kpts)

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

