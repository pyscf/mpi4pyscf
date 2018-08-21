#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import platform
import ctypes
import tempfile
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
from pyscf.pbc import gto as pbcgto

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
        if isinstance(mydf._cderi_to_save, str):
            cderi = mydf._cderi_to_save
        else:
            cderi = mydf._cderi_to_save.name
        if isinstance(mydf._cderi, str):
            log.warn('Value of _cderi is ignored. DF integrals will be '
                     'saved in file %s .', cderi)
        mydf._cderi = cderi
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
    nauxs = []
    t1 = log.timer_debug1('2c2e', *t1)

    if h5py.is_hdf5(cderi_file):
        feri = h5py.File(cderi_file)
    else:
        feri = h5py.File(cderi_file, 'w')

    if cell.dimension == 1 or cell.dimension == 2:
        plain_ints = df._gaussian_int(fused_cell)

    mem_now = max(comm.allgather(lib.current_memory()[0]))
    max_memory = max(2000, mydf.max_memory - mem_now)
    blksize = max(2048, int(max_memory*.5e6/16/fused_cell.nao_nr()))
    log.debug2('max_memory %s (MB)  blocksize %s', max_memory, blksize)
    for k, kpt in enumerate(uniq_kpts):
        coulG = numpy.sqrt(mydf.weighted_coulG(kpt, False, mesh))
        j2c_k = numpy.zeros_like(j2c[k])
        for p0, p1 in mydf.prange(0, ngrids, blksize):
            aoaux = ft_ao.ft_ao(fused_cell, Gv[p0:p1], None, b, gxyz[p0:p1], Gvbase, kpt)
            if (cell.dimension == 1 or cell.dimension == 2) and is_zero(kpt):
                G0idx, SI_on_z = pbcgto.cell._SI_for_uniform_model_charge(cell, Gv[p0:p1])
                aoaux[G0idx] -= numpy.einsum('g,i->gi', SI_on_z, plain_ints)
            aoaux = aoaux.T
            LkR = aoaux.real * coulG[p0:p1]
            LkI = aoaux.imag * coulG[p0:p1]
            aoaux = None

            if is_zero(kpt):  # kpti == kptj
                j2c_k[naux:] += lib.ddot(LkR[naux:], LkR.T)
                j2c_k[naux:] += lib.ddot(LkI[naux:], LkI.T)
            else:
                j2cR, j2cI = zdotCN(LkR[naux:], LkI[naux:], LkR.T, LkI.T)
                j2c_k[naux:] += j2cR + j2cI * 1j
            kLR = kLI = None

        j2c_k[:naux,naux:] = j2c_k[naux:,:naux].conj().T
        j2c[k] -= mpi.allreduce(j2c_k)
        j2c[k] = fuse(fuse(j2c[k]).T).T
        try:
            feri['j2c/%d'%k] = scipy.linalg.cholesky(j2c[k], lower=True)
            j2ctags.append('CD')
            nauxs.append(naux)
        except scipy.linalg.LinAlgError as e:
            #msg =('===================================\n'
            #      'J-metric not positive definite.\n'
            #      'It is likely that mesh is not enough.\n'
            #      '===================================')
            #log.error(msg)
            #raise scipy.linalg.LinAlgError('\n'.join([e.message, msg]))
            w, v = scipy.linalg.eigh(j2c[k])
            log.debug2('metric linear dependency for kpt %s', k)
            log.debug2('cond = %.4g, drop %d bfns',
                       w[0]/w[-1], numpy.count_nonzero(w<mydf.linear_dep_threshold))
            v = v[:,w>mydf.linear_dep_threshold].T.conj()
            v /= numpy.sqrt(w[w>mydf.linear_dep_threshold]).reshape(-1,1)
            feri['j2c/%d'%k] = v
            j2ctags.append('eig')
            nauxs.append(v.shape[0])
    j2c = coulG = None

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

    j3c_jobs = grids2d_int3c_jobs(cell, auxcell, kptij_lst, chunks, j_only)
    log.debug1('max_memory = %d MB (%d in use)  chunks %s',
               max_memory, mem_now, chunks)
    log.debug2('j3c_jobs %s', j3c_jobs)

    if j_only:
        int3c = wrap_int3c(cell, fused_cell, 'int3c2e', 's2', 1, kptij_lst)
    else:
        int3c = wrap_int3c(cell, fused_cell, 'int3c2e', 's1', 1, kptij_lst)
        idxb = numpy.tril_indices(nao)
        idxb = (idxb[0] * nao + idxb[1]).astype('i')
    aux_loc = fused_cell.ao_loc_nr('ssc' in 'int3c2e')

    def gen_int3c(auxcell, job_id, ish0, ish1):
        dataname = 'j3c-chunks/%d' % job_id
        if dataname in feri:
            del(feri[dataname])

        i0 = ao_loc[ish0]
        i1 = ao_loc[ish1]
        dii = i1*(i1+1)//2 - i0*(i0+1)//2
        if j_only:
            dij = dii
            buflen = max(8, int(max_memory*1e6/16/(nkptij*dii+dii)))
        else:
            dij = (i1 - i0) * nao
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
            log.alldebug2("aux_e1 job_id %d step %d", job_id, istep)
            sh0, sh1, nrow = auxrange
            sub_slice = (ish0, ish1, 0, cell.nbas, sh0, sh1)
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

        j2c = numpy.asarray(feri['j2c/%d'%uniq_kptji_id])
        j2ctag = j2ctags[uniq_kptji_id]
        naux0 = j2c.shape[0]

        if is_zero(kpt):
            aosym = 's2'
            vbar = mydf.auxbar(fused_cell)
            ovlp = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=adapted_kptjs)
            ovlp = [lib.pack_tril(s) for s in ovlp]
        else:
            aosym = 's1'

        j3cR = [None] * nkptj
        j3cI = [None] * nkptj
        i0 = ao_loc[sh0]
        i1 = ao_loc[sh1]
        for k, idx in enumerate(adapted_ji_idx):
            key = 'j3c-chunks/%d/%d' % (job_id, idx)
            v = numpy.asarray(feri[key])
            if is_zero(kpt) and cell.dimension == 3:
                for i, c in enumerate(vbar):
                    if c != 0:
                        v[i] -= c * ovlp[k][i0*(i0+1)//2:i1*(i1+1)//2].ravel()
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
        log.alldebug2('job_id %d  blksize (%d,%d)', job_id, Gblksize, ncol)

        wcoulG = mydf.weighted_coulG(kpt, False, mesh)
        fused_cell_slice = (auxcell.nbas, fused_cell.nbas)
        if aosym == 's2':
            shls_slice = (sh0, sh1, 0, sh1)
        else:
            shls_slice = (sh0, sh1, 0, cell.nbas)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            Gaux = ft_ao.ft_ao(fused_cell, Gv[p0:p1], fused_cell_slice, b,
                               gxyz[p0:p1], Gvbase, kpt)
            Gaux *= wcoulG[p0:p1,None]
            kLR = Gaux.real.copy('C')
            kLI = Gaux.imag.copy('C')
            Gaux = None

            dat = ft_ao._ft_aopair_kpts(cell, Gv[p0:p1], shls_slice, aosym, b,
                                        gxyz[p0:p1], Gvbase, kpt,
                                        adapted_kptjs, out=buf)

            if (cell.dimension == 1 or cell.dimension == 2) and is_zero(kpt):
                G0idx, SI_on_z = pbcgto.cell._SI_for_uniform_model_charge(cell, Gv[p0:p1])
                if SI_on_z.size > 0:
                    for k, aoao in enumerate(dat):
                        aoao[G0idx] -= numpy.einsum('g,i->gi', SI_on_z, ovlp[k])
                        aux = fuse(ft_ao.ft_ao(fused_cell, Gv[p0:p1][G0idx]).T)
                        vG_mod = numpy.einsum('ig,g,g->i', aux.conj(),
                                              wcoulG[p0:p1][G0idx], SI_on_z)
                        if gamma_point(adapted_kptjs[k]):
                            j3cR[k][:naux] -= vG_mod[:,None].real * ovlp[k]
                        else:
                            tmp = vG_mod[:,None] * ovlp[k]
                            j3cR[k][:naux] -= tmp.real
                            j3cI[k][:naux] -= tmp.imag
                        tmp = aux = vG_mod

            nG = p1 - p0
            for k, ji in enumerate(adapted_ji_idx):
                aoao = dat[k].reshape(nG,ncol)
                pqkR = numpy.ndarray((ncol,nG), buffer=pqkRbuf)
                pqkI = numpy.ndarray((ncol,nG), buffer=pqkIbuf)
                pqkR[:] = aoao.real.T
                pqkI[:] = aoao.imag.T

                lib.dot(kLR.T, pqkR.T, -1, j3cR[k][naux:], 1)
                lib.dot(kLI.T, pqkI.T, -1, j3cR[k][naux:], 1)
                if not (is_zero(kpt) and gamma_point(adapted_kptjs[k])):
                    lib.dot(kLR.T, pqkI.T, -1, j3cI[k][naux:], 1)
                    lib.dot(kLI.T, pqkR.T,  1, j3cI[k][naux:], 1)
            kLR = kLI = None

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
    fuse = gen_int3c = ft_fuse = None
    t1 = log.timer_debug1('int3c and fuse', *t1)

    def get_segs_loc(aosym):
        off0 = numpy.asarray([ao_loc[i0] for x,i0,i1 in j3c_jobs])
        off1 = numpy.asarray([ao_loc[i1] for x,i0,i1 in j3c_jobs])
        if aosym:  # s2
            dims = off1*(off1+1)//2 - off0*(off0+1)//2
        else:
            dims = (off1-off0) * nao
        #dims = numpy.asarray([ao_loc[i1]-ao_loc[i0] for x,i0,i1 in j3c_jobs])
        dims = numpy.hstack([dims[j3c_workers==w] for w in range(mpi.pool.size)])
        job_idx = numpy.hstack([numpy.where(j3c_workers==w)[0]
                                for w in range(mpi.pool.size)])
        segs_loc = numpy.append(0, numpy.cumsum(dims))
        segs_loc = [(segs_loc[j], segs_loc[j+1]) for j in numpy.argsort(job_idx)]
        return segs_loc
    segs_loc_s1 = get_segs_loc(False)
    segs_loc_s2 = get_segs_loc(True)

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
            if aosym_s2[k]:
                segs = numpy.hstack([segs[i0*nL:i1*nL].reshape(nL,-1)
                                     for i0,i1 in segs_loc_s2])
            else:
                segs = numpy.hstack([segs[i0*nL:i1*nL].reshape(nL,-1)
                                     for i0,i1 in segs_loc_s1])
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
    with lib.call_in_background(save) as async_write:
        for k, kptji in enumerate(kptij_lst):
            for p0, p1 in lib.prange(0, segsize, blksize):
                segs = load(k, p0, p1)
                async_write(k, p0, p1, segs)
                t2 = log.timer_debug1('assemble k=%d %d:%d (in %d)' %
                                      (k, p0, p1, segsize), *t2)

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
        return df.DF.dump_flags(self, logger.Logger(self.stdout, self.verbose))

    def pack(self):
        return {'verbose'   : self.verbose,
                'max_memory': self.max_memory,
                'kpts'      : self.kpts,
                'kpts_band' : self.kpts_band,
                'mesh'      : self.mesh,
                'eta'       : self.eta,
                'exp_to_discard' : self.exp_to_discard,
                'blockdim'  : self.blockdim,
                '_auxbasis' : self._auxbasis,
                'linear_dep_threshold': self.linear_dep_threshold,
                '_cderi'     : self._cderi}
    def unpack_(self, dfdic):
        remote_cderi = dfdic.pop('_cderi')
        self.__dict__.update(dfdic)
# Note when auxbasis was changed in the master process, _cderi on master is
# cleared.  Following to reset _cderi and _cderi_to_save when necessary.
        if remote_cderi is None and self._cderi is not None:
            self._cderi = None
            self._cderi_to_save = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
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

    mpi_prange = prange = aft.AFTDF.mpi_prange


    def loop(self):
        # mpi.pool.worker_status = P (pending) means the caller on master
        # process runs in serial mode.
        # 3-index tensor should be generated on each process and sent to
        # the master process.  E.g. the call to with_df.loop in dfccsd
        serial_mode = mpi.pool.worker_status == 'P'
        if serial_mode:
            return loop_yield_then_reduce(self)
        else:
            return df.DF.loop(self)

    def get_naoaux(self):
        if mpi.pool.worker_status == 'P':
            return get_naoaux(self)
        else:
            return df.DF.get_naoaux(self)


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

@mpi.reduced_yield
def loop_yield_then_reduce(mydf):
    for Lpq in df.DF.loop(mydf):
        yield Lpq

@mpi.call_then_reduce
def get_naoaux(mydf):
    return df.DF.get_naoaux(mydf)


if __name__ == '__main__':
    from pyscf.pbc import gto as pgto
    from mpi4pyscf.pbc import df
    cell = pgto.M(atom='He 0 0 0; He 0 0 1', a=numpy.eye(3)*4, mesh=[11]*3)
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

