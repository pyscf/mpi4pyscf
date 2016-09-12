#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Exact density fitting with Gaussian and planewaves
Ref:
'''

import time
import platform
import copy
import ctypes
import numpy
import scipy.linalg
import h5py

from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc import tools
from pyscf.pbc.df import incore
from pyscf.pbc.df import outcore
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import mdf
from pyscf.pbc.df.mdf_jk import zdotNN, zdotCN, zdotNC
from pyscf.gto.mole import PTR_COORD
from pyscf.ao2mo.outcore import balance_segs

from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi
from mpi4pyscf.pbc.df import mdf_jk
from mpi4pyscf.pbc.df import mdf_ao2mo
from mpi4pyscf.pbc.df import pwdf

comm = mpi.comm
rank = mpi.rank


def init_MDF(cell, kpts=numpy.zeros((1,3))):
    mydf = mpi.pool.apply(_init_MDF_wrap, [cell, kpts], [cell.dumps(), kpts])
    return mydf
def _init_MDF_wrap(args):
    from mpi4pyscf.pbc.df import mdf
    cell, kpts = args
    if mdf.rank > 0:
        cell = mdf.gto.loads(cell)
        cell.verbose = 0
    return mdf.mpi.register_for(mdf.MDF(cell, kpts))

def get_nuc(mydf, kpts=None):
    if mydf._cderi is None:
        mydf.build()
    args = (mydf._reg_keys, kpts)
    return mpi.pool.apply(_get_nuc_wrap, args, args)
def _get_nuc_wrap(args):
    from mpi4pyscf.pbc.df import mdf
    return mdf._get_nuc(*args)
def _get_nuc(reg_keys, kpts=None):
    mydf = mdf_jk._load_df(reg_keys)
    cell = mydf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = t0 = (time.clock(), time.time())
    nao = cell.nao_nr()
    auxcell = mydf.auxcell
    nuccell = mdf.make_modchg_basis(cell, mydf.eta, 0)
    nuccell._bas = numpy.asarray(nuccell._bas[nuccell._bas[:,gto.mole.ANG_OF]==0],
                                 dtype=numpy.int32, order='C')
    charge = -cell.atom_charges()
    nucbar = sum([z/nuccell.bas_exp(i)[0] for i,z in enumerate(charge)])
    nucbar *= numpy.pi/cell.vol

    vj = _int_nuc_vloc(cell, nuccell, kpts_lst)
    vj = vj.reshape(-1,nao**2)
# Note j2c may break symmetry
    j2c = gto.intor_cross('cint2c2e_sph', auxcell, nuccell)
    jaux = j2c.dot(charge)
    t1 = log.timer_debug1('vnuc pass1: analytic int', *t1)

    kpt_allow = numpy.zeros(3)
    coulG = tools.get_coulG(cell, kpt_allow, gs=mydf.gs) / cell.vol
    Gv = cell.get_Gv(mydf.gs)
    aoaux = ft_ao.ft_ao(nuccell, Gv)
    vGR = numpy.einsum('i,xi->x', charge, aoaux.real) * coulG
    vGI = numpy.einsum('i,xi->x', charge, aoaux.imag) * coulG

    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
    for k, pqkR, pqkI, p0, p1 \
            in mydf.ft_loop(cell, mydf.gs, kpt_allow, kpts_lst, max_memory=max_memory):
# rho_ij(G) nuc(-G) / G^2
# = [Re(rho_ij(G)) + Im(rho_ij(G))*1j] [Re(nuc(G)) - Im(nuc(G))*1j] / G^2
        if not gamma_point(kpts_lst[k]):
            vj[k] += numpy.einsum('k,xk->x', vGR[p0:p1], pqkI) * 1j
            vj[k] += numpy.einsum('k,xk->x', vGI[p0:p1], pqkR) *-1j
        vj[k] += numpy.einsum('k,xk->x', vGR[p0:p1], pqkR)
        vj[k] += numpy.einsum('k,xk->x', vGI[p0:p1], pqkI)
    t1 = log.timer_debug1('contracting Vnuc', *t1)

    vj = vj.reshape(-1,nao,nao)
# Append nuccell to auxcell, so that they can be FT together in pw_loop
# the first [:naux] of ft_ao are aux fitting functions.
    nuccell._atm, nuccell._bas, nuccell._env = \
            gto.conc_env(auxcell._atm, auxcell._bas, auxcell._env,
                         nuccell._atm, nuccell._bas, nuccell._env)
    naux = auxcell.nao_nr()
    aoaux = ft_ao.ft_ao(nuccell, Gv)
    vG = numpy.einsum('i,xi,x->x', charge, aoaux[:,naux:], coulG)
    jaux -= numpy.einsum('x,xj->j', vG.real, aoaux[:,:naux].real)
    jaux -= numpy.einsum('x,xj->j', vG.imag, aoaux[:,:naux].imag)
    jaux -= charge.sum() * mydf.auxbar(auxcell)
    jobs = mpi.static_partition(range(naux))
    jaux = jaux[jobs]

    nao_pair = nao * (nao+1) // 2
    max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
    blksize = max(16, min(int(max_memory*1e6/16/nao_pair), mydf.blockdim))

    for k, kpt in enumerate(kpts_lst):
        with mydf.load_Lpq((kpt,kpt)) as Lpq:
            nrow = Lpq.shape[0]
            if nrow > 0:
                v = 0
                for p0, p1 in lib.prange(0, nrow, blksize):
                    v += numpy.dot(jaux[p0:p1], numpy.asarray(Lpq[p0:p1]))
                vj[k] += lib.unpack_tril(v)

    vj = mpi.reduce(lib.asarray(vj))
    if rank == 0:
        ovlp = cell.pbc_intor('cint1e_ovlp_sph', 1, lib.HERMITIAN, kpts_lst)
        nuc = []
        for k, kpt in enumerate(kpts_lst):
            if gamma_point(kpt):
                nuc.append(vj[k].real - nucbar * ovlp[k].real)
            else:
                nuc.append(vj[k] - nucbar * ovlp[k])
        if kpts is None or numpy.shape(kpts) == (3,):
            nuc = nuc[0]
        return nuc


def get_pp(mydf, kpts=None):
    cell = mydf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))
    nkpts = len(kpts_lst)

    vloc1 = get_nuc(mydf, kpts_lst)
    vloc2 = gto.pseudo.pp_int.get_pp_loc_part2(cell, kpts_lst)
    vpp = gto.pseudo.pp_int.get_pp_nl(cell, kpts_lst)
    for k in range(nkpts):
        vpp[k] += vloc1[k] + vloc2[k]

    if kpts is None or numpy.shape(kpts) == (3,):
        vpp = vpp[0]
    return vpp


class MDF(mdf.MDF, pwdf.PWDF):

    def build(self, j_only=False, with_j3c=True):
        args = (self._reg_keys, j_only, with_j3c)
        mpi.pool.apply(_build_wrap, args, args)
        return self

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

        if kpts.shape == (3,):
            return mdf_jk.get_jk(self, dm, hermi, kpts, kpt_band, with_j,
                                 with_k, exxdiv)

        vj = vk = None
        if with_k:
            vk = mdf_jk.get_k_kpts(self, dm, hermi, kpts, kpt_band, exxdiv)
        if with_j:
            vj = mdf_jk.get_j_kpts(self, dm, hermi, kpts, kpt_band)
        return vj, vk

    get_eri = get_ao_eri = mdf_ao2mo.get_eri
    ao2mo = get_mo_eri = mdf_ao2mo.general


def _build_wrap(args):
    from mpi4pyscf.pbc.df import mdf
    return mdf._build(*args)
def _build(reg_keys, j_only=False, with_j3c=True):
# Unlike DF and PWDF class, here MDF objects are synced once
    mydf = mpi._registry[reg_keys[rank]]
    if mpi.pool.size == 1:
        return mdf.MDF.build(mydf, j_only, with_j3c)

    mydf.kpts, mydf.gs, mydf.metric, mydf.approx_sr_level, mydf.auxbasis, \
            mydf.eta, mydf.exxdiv, mydf.blockdim = \
            comm.bcast((mydf.kpts, mydf.gs, mydf.metric, mydf.approx_sr_level,
                        mydf.auxbasis, mydf.eta, mydf.exxdiv, mydf.blockdim))

    log = logger.Logger(mydf.stdout, mydf.verbose)
    info = rank, platform.node(), platform.os.getpid()
    log.debug('MPI info (rank, host, pid)  %s', comm.gather(info))

    t1 = (time.clock(), time.time())
    cell = mydf.cell
    if mydf.eta is None:
        mydf.eta = mdf.estimate_eta(cell)
        log.debug('Set smooth gaussian eta to %.9g', mydf.eta)
    mydf.dump_flags()

    auxcell = mdf.make_modrho_basis(cell, mydf.auxbasis, mydf.eta)
    chgcell = mdf.make_modchg_basis(auxcell, mydf.eta)

    mydf._j_only = j_only
    if j_only:
        kptij_lst = numpy.hstack((mydf.kpts,mydf.kpts)).reshape(-1,2,3)
    else:
        kptij_lst = [(ki, mydf.kpts[j])
                     for i, ki in enumerate(mydf.kpts) for j in range(i+1)]
        kptij_lst = numpy.asarray(kptij_lst)

    if not isinstance(mydf._cderi, str):
        if isinstance(mydf._cderi_file, str):
            mydf._cderi = mydf._cderi_file
        else:
            mydf._cderi = mydf._cderi_file.name

    if with_j3c:
        if mydf.approx_sr_level != 0:
            raise NotImplementedError

        _make_j3c(mydf, cell, auxcell, chgcell, kptij_lst)
        t1 = log.timer_debug1('3c2e', *t1)
    else:
        raise NotImplementedError

# Merge chgcell into auxcell
    auxcell._atm, auxcell._bas, auxcell._env = \
            gto.conc_env(auxcell._atm, auxcell._bas, auxcell._env,
                         chgcell._atm, chgcell._bas, chgcell._env)
    mydf.auxcell = auxcell
    return mydf

def _make_j3c(mydf, cell, auxcell, chgcell, kptij_lst):
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = t0 = (time.clock(), time.time())
    auxcell, auxcell_short = copy.copy(auxcell), auxcell
    auxcell._atm, auxcell._bas, auxcell._env = \
            gto.conc_env(auxcell._atm, auxcell._bas, auxcell._env,
                         chgcell._atm, chgcell._bas, chgcell._env)
    ao_loc = cell.ao_loc_nr()
    aux_loc = auxcell.ao_loc_nr()
    nao = ao_loc[-1]
    naux = aux_loc[-1]
    naux_short = auxcell_short.nao_nr()
    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    kpt_ji = kptjs - kptis
    uniq_kpts, uniq_index, uniq_inverse = mdf.unique(kpt_ji)

    gs = mydf.gs
    gxyz = lib.cartesian_prod((numpy.append(range(gs[0]+1), range(-gs[0],0)),
                               numpy.append(range(gs[1]+1), range(-gs[1],0)),
                               numpy.append(range(gs[2]+1), range(-gs[2],0))))
    invh = numpy.linalg.inv(cell._h)
    Gv = 2*numpy.pi * numpy.dot(gxyz, invh)
    ngs = gxyz.shape[0]
    # j2c ~ (-kpt_ji | kpt_ji)
    j2c = auxcell.pbc_intor('cint2c2e_sph', hermi=1, kpts=uniq_kpts)
    t1 = log.timer_debug1('2c2e', *t1)
    kLRs = []
    kLIs = []
    for k, kpt in enumerate(uniq_kpts):
        aoaux = ft_ao.ft_ao(auxcell, Gv, None, invh, gxyz, gs, kpt)
        coulG = numpy.sqrt(tools.get_coulG(cell, kpt, gs=gs) / cell.vol)
        kLR = aoaux.real * coulG.reshape(-1,1)
        kLI = aoaux.imag * coulG.reshape(-1,1)
        if not kLR.flags.c_contiguous: kLR = lib.transpose(kLR.T)
        if not kLI.flags.c_contiguous: kLI = lib.transpose(kLI.T)
        aoaux = None

        for p0, p1 in mydf.mpi_prange(0, ngs):
            if is_zero(kpt):  # kpti == kptj
                j2cR = lib.dot(kLR[p0:p1].T, kLR[p0:p1])
                j2cR = lib.dot(kLI[p0:p1].T, kLI[p0:p1], 1, j2cR, 1)
                j2c[k] -= mpi.allreduce(j2cR)
            else:
                 # aoaux ~ kpt_ij, aoaux.conj() ~ kpt_kl
                j2cR, j2cI = zdotCN(kLR[p0:p1], kLI[p0:p1], kLR[p0:p1], kLI[p0:p1])
                j2cR = mpi.allreduce(j2cR)
                j2cI = mpi.allreduce(j2cI)
                j2c[k] -= j2cR + j2cI * 1j
        j2cR = j2cI = None

        kLR *= coulG.reshape(-1,1)
        kLI *= coulG.reshape(-1,1)
        kLRs.append(kLR)
        kLIs.append(kLI)
    t1 = log.timer_debug1('aoaux', *t1)

# Estimates the buffer size based on the last contraction in G-space.
# This contraction requires to hold nkptj copies of (naux,?) array
# simultaneously in memory.
    max_memory = mydf.max_memory - lib.current_memory()[0]
    max_memory = max(2000, min(comm.allgather(max_memory)))
    #nkptj_max = max(numpy.unique(uniq_inverse, return_counts=True)[1])
    nkptj_max = max((uniq_inverse==x).sum() for x in set(uniq_inverse))
    buflen = int(min(max(max_memory*.5e6/16/naux/(nkptj_max+1)/nao, 1),
                     nao/mpi.pool.size+3, nao))
    chunks = (buflen, nao)

    j3c_jobs = grids2d_int3c_jobs(cell, auxcell_short, kptij_lst, chunks)
    log.debug1('max_memory = %s MB  chunks %s', max_memory, chunks)
    log.debug2('j3c_jobs %s', j3c_jobs)

    if mydf.metric.upper() == 'S':
        _aux_e2(cell, auxcell_short, mydf._cderi, kptij_lst, j3c_jobs,
                'cint3c1e_sph', 'Lpq-chunks', max_memory)
        s_aux = auxcell_short.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=uniq_kpts)
    else:
        _aux_e2(cell, auxcell_short, mydf._cderi, kptij_lst, j3c_jobs,
                'cint3c1e_p2_sph', 'Lpq-chunks', max_memory)
        s_aux = auxcell_short.pbc_intor('cint1e_kin_sph', hermi=1, kpts=uniq_kpts)
        s_aux = [x*2 for x in s_aux]
    s_aux = [scipy.linalg.cho_factor(x) for x in s_aux]
    compress = mdf.compress_Lpq_to_chgcell(auxcell_short, chgcell)
    t1 = log.timer_debug1('Lpq', *t1)

    j3c_workers = _aux_e2(cell, auxcell, mydf._cderi, kptij_lst, j3c_jobs,
                          'cint3c2e_sph', 'j3c-chunks', max_memory)
    log.debug2('j3c_workers %s', j3c_workers)
    t1 = log.timer_debug1('3c2e', *t1)
    ####

    feri = h5py.File(mydf._cderi)
    def fuse(Lpq, j3c, uniq_k_id, key):
        Lpq = compress(scipy.linalg.cho_solve(s_aux[uniq_k_id], Lpq))
        del(feri['Lpq'+key])
        feri['Lpq'+key] = Lpq
        feri['j3c'+key][:] = lib.dot(j2c[uniq_k_id], Lpq, -.5, j3c, 1)

    write_handler = None
    for i, jobs in enumerate(j3c_jobs):
        if j3c_workers[i] == rank:
            job_id = jobs[0]
            for k, kptij in enumerate(kptij_lst):
                uniq_k_id = uniq_inverse[k]
                key = '-chunks/%d/%d' % (job_id, k)
                Lpq = numpy.asarray(feri['Lpq'+key])
                j3c = numpy.asarray(feri['j3c'+key])
                write_handler = async_write(write_handler, fuse, Lpq, j3c,
                                            uniq_k_id, key)
                Lpq = j3c = None
    write_handler.join()
    write_handler = coulG = s_aux = j2c = compress = None
    t1 = log.timer_debug1('distributing Lpq j3c', *t1)
    ####

    aosym_s2 = numpy.einsum('ix->i', abs(kptis-kptjs)) < 1e-9
    vbar = mydf.auxbar(auxcell)
    ovlp = cell.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=kptjs[aosym_s2])

    def process(job_id, uniq_kptji_id, sh0, sh1):
        log.debug1("job_id %d uniq_kptji_id %d", job_id, uniq_kptji_id)
        kpt = uniq_kpts[uniq_kptji_id]  # kpt = kptj - kpti
        adapted_ji_idx = numpy.where(uniq_inverse == uniq_kptji_id)[0]
        adapted_kptjs = kptjs[adapted_ji_idx]
        nkptj = len(adapted_kptjs)
        kLR = kLRs[uniq_kptji_id]
        kLI = kLIs[uniq_kptji_id]

        j3cR = []
        j3cI = []
        for idx in adapted_ji_idx:
            v = numpy.asarray(feri['j3c-chunks/%d/%d'%(job_id,idx)])
            j3cR.append(numpy.asarray(v.real, order='C'))
            if v.dtype == numpy.complex128:
                j3cI.append(numpy.asarray(v.imag, order='C'))
            else:
                j3cI.append(None)
            v = None

        max_memory = max(2000, mydf.max_memory - lib.current_memory()[0])
        ncol = j3cR[0].shape[1]
        Gblksize = max(16, int(max_memory*1e6/16/ncol/(nkptj+1)))  # +1 for pqkRbuf/pqkIbuf
        Gblksize = min(Gblksize, ngs)
        pqkRbuf = numpy.empty(ncol*Gblksize)
        pqkIbuf = numpy.empty(ncol*Gblksize)
        # buf for ft_aopair
        buf = numpy.zeros((nkptj,ncol*Gblksize), dtype=numpy.complex128)

        shls_slice = (sh0, sh1, 0, cell.nbas)
        ni = ncol // nao
        for p0, p1 in lib.prange(0, ngs, Gblksize):
            ft_ao._ft_aopair_kpts(cell, Gv[p0:p1], shls_slice, 's1', invh,
                                  gxyz[p0:p1], gs, kpt, adapted_kptjs, out=buf)
            nG = p1 - p0
            for k, ji in enumerate(adapted_ji_idx):
                aoao = numpy.ndarray((nG,ni,nao), dtype=numpy.complex128,
                                     order='F', buffer=buf[k])
                pqkR = numpy.ndarray((ni,nao,nG), buffer=pqkRbuf)
                pqkI = numpy.ndarray((ni,nao,nG), buffer=pqkIbuf)
                pqkR[:] = aoao.real.transpose(1,2,0)
                pqkI[:] = aoao.imag.transpose(1,2,0)
                aoao[:] = 0
                pqkR = pqkR.reshape(-1,nG)
                pqkI = pqkI.reshape(-1,nG)

                if is_zero(kpt):  # kpti == kptj
                    # *.5 for hermi_sum at the assemble step
                    if gamma_point(adapted_kptjs[k]):
                        lib.dot(kLR[p0:p1].T, pqkR.T, -.5, j3cR[k], 1)
                        lib.dot(kLI[p0:p1].T, pqkI.T, -.5, j3cR[k], 1)
                    else:
                        zdotCN(kLR[p0:p1].T, kLI[p0:p1].T, pqkR.T, pqkI.T,
                               -.5, j3cR[k], j3cI[k], 1)
                else:
                    zdotCN(kLR[p0:p1].T, kLI[p0:p1].T, pqkR.T, pqkI.T,
                           -1, j3cR[k], j3cI[k], 1)

        if is_zero(kpt):
            i0 = ao_loc[sh0]
            i1 = ao_loc[sh1]
            for k, ji in enumerate(adapted_ji_idx):
                if is_zero(adapted_kptjs[k]):
                    for i, c in enumerate(vbar):
                        if c != 0:
                            j3cR[k][i] -= c*.5 * ovlp[k][i0:i1].real.ravel()
                else:
                    for i, c in enumerate(vbar):
                        if c != 0:
                            j3cR[k][i] -= c*.5 * ovlp[k][i0:i1].real.ravel()
                            j3cI[k][i] -= c*.5 * ovlp[k][i0:i1].imag.ravel()

        for k, idx in enumerate(adapted_ji_idx):
            if is_zero(kpt) and gamma_point(adapted_kptjs[k]):
                feri['j3c-chunks/%d/%d'%(job_id,idx)][:] = j3cR[k]
            else:
                feri['j3c-chunks/%d/%d'%(job_id,idx)][:] = j3cR[k] + j3cI[k]*1j

    def fuse():
        for i, jobs in enumerate(j3c_jobs):
            if j3c_workers[i] == rank:
                job_id, sh0, sh1 = jobs
                for k, kpt in enumerate(uniq_kpts):
                    process(job_id, k, sh0, sh1)

    if 'Lpq' in feri: del(feri['Lpq'])
    if 'j3c' in feri: del(feri['j3c'])
    segsize = (naux+mpi.pool.size-1) // mpi.pool.size
    naux0 = min(naux, rank*segsize)
    naux1 = min(naux, rank*segsize+segsize)
    nrow = naux1 - naux0
    for k, kptij in enumerate(kptij_lst):
        if gamma_point(kptij):
            dtype = 'f8'
        else:
            dtype = 'c16'
        if aosym_s2[k]:
            nao_pair = nao * (nao+1) // 2
        else:
            nao_pair = nao * nao
        feri.create_dataset('Lpq/%d'%k, (nrow,nao_pair), dtype, maxshape=(None,nao_pair))
        feri.create_dataset('j3c/%d'%k, (nrow,nao_pair), dtype, maxshape=(None,nao_pair))

    dims = numpy.asarray([ao_loc[i1]-ao_loc[i0] for x,i0,i1 in j3c_jobs])
    dims = [dims[j3c_workers==w].sum() * nao for w in range(mpi.pool.size)]
    def load(label, k, p0, p1):
        slices = [(min(i*segsize+p0,naux), min(i*segsize+p1, naux))
                  for i in range(mpi.pool.size)]
        segs = []
        for p0, p1 in slices:
            val = []
            for job_id, worker in enumerate(j3c_workers):
                if rank == worker:
                    key = '-chunks/%d/%d' % (job_id, k)
                    val.append(feri[label+key][p0:p1])
            if val:
                segs.append(numpy.hstack(val))
            else:
                segs.append(numpy.zeros(0))
        return segs

    def save(label, k, p0, p1, segs):
        loc0, loc1 = min(p0, naux-naux0), min(p1, naux-naux0)
        segs = mpi.alltoall(segs, True)
        segs = numpy.hstack([x.reshape(-1,dims[i]) for i,x in enumerate(segs)])
        if aosym_s2[k]:
            segs = lib.hermi_sum(segs.reshape(-1,nao,nao), axes=(0,2,1))
            segs = lib.pack_tril(segs)
        feri['%s/%d'%(label,k)][loc0:loc1] = segs

    max_memory = min(8000, mydf.max_memory - lib.current_memory()[0])
    max_memory = max(2000, min(comm.allgather(max_memory)))
    if numpy.all(aosym_s2):
        if gamma_point(kptij_lst):
            blksize = max(16, int(max_memory*1e6/8/nao**2))
        else:
            blksize = max(16, int(max_memory*1e6/16/nao**2))
    else:
        blksize = max(16, int(max_memory*1e6/16/nao**2/2))

    thread_fusion = lib.background_thread(fuse)

    t2 = t1
    write_handler = None
    for k, kptji in enumerate(kptij_lst):
        for p0, p1 in lib.prange(0, segsize, blksize):
            segs = load('Lpq', k, p0, p1)
            write_handler = async_write(write_handler, save, 'Lpq', k, p0, p1, segs)
            t2 = log.timer_debug1('assemble Lpq k=%d %d:%d (in %d)' % (k, p0, p1, nrow), *t2)
            segs = None
    write_handler.join()
    write_handler = None

    thread_fusion.join()
    kLRs = kLIs = ovlp = None
    t1 = log.timer_debug1('fusing Lpq j3c', *t1)

    t2 = t1
    write_handler = None
    for k, kptji in enumerate(kptij_lst):
        for p0, p1 in lib.prange(0, segsize, blksize):
            segs = load('j3c', k, p0, p1)
            write_handler = async_write(write_handler, save, 'j3c', k, p0, p1, segs)
            t2 = log.timer_debug1('assemble k=%d %d:%d (in %d)' % (k, p0, p1, nrow), *t2)
            segs = None
    write_handler.join()
    write_handler = None

    if 'Lpq-chunks' in feri: del(feri['Lpq-chunks'])
    if 'j3c-chunks' in feri: del(feri['j3c-chunks'])
    t1 = log.timer_debug1('assembling Lpq j3c', *t1)

    if 'Lpq-kptij' in feri: del(feri['Lpq-kptij'])
    if 'Lpq-kptij' in feri: del(feri['j3c-kptij'])
    feri['Lpq-kptij'] = kptij_lst
    feri['j3c-kptij'] = kptij_lst
    feri.close()

def grids2d_int3c_jobs(cell, auxcell, kptij_lst, chunks):
    ao_loc = cell.ao_loc_nr()
    nao = ao_loc[-1]
    segs = ao_loc[1:]-ao_loc[:-1]
    ij_ranges = balance_segs(segs, chunks[0])

    jobs = [(job_id, i0, i1) for job_id, (i0, i1, x) in enumerate(ij_ranges)]
    return jobs

# Note on each proccessor, _int_nuc_vloc computes only a fraction of the entire vj.
# It is because the summation over real space images are splited by mpi.static_partition
def _int_nuc_vloc(cell, nuccell, kpts):
    '''Vnuc - Vloc'''
    nimgs = numpy.max((cell.nimgs, nuccell.nimgs), axis=0)
    Ls = numpy.asarray(cell.get_lattice_Ls(nimgs), order='C')
    expLk = numpy.asarray(numpy.exp(1j*numpy.dot(Ls, kpts.T)), order='C')
    nkpts = len(kpts)

    fakenuc = mdf._fake_nuc(cell)
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
    ints = incore._wrap_int3c(cell, fakenuc, 'cint3c2e_sph', 1, Ls, buf)
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
# nuc is mpi.reduced in get_nuc function
    return nuc

def _assign_kpts_task(kpts, kptij_lst):
    mask = numpy.zeros(len(kptij_lst), dtype=bool)
    kpts_ji = kptij_lst[:,1] - kptij_lst[:,0]
    uniq_kpts, uniq_index, uniq_inverse = mdf.unique(kpts_ji)

    for kpt in mpi.static_partition(uniq_kptji):
        mask |= numpy.einsum('ki->k', abs(kpts_ji-kpt)) < 1e-9

    worker = numpy.zeros(len(kptij_lst), dtype=int)
    worker[mask] = rank
    worker = mpi.allreduce(worker)
    return worker

# Compute only half of the integrals for aosym='s2'
def _aux_e2(cell, auxcell, erifile, kptij_lst, all_jobs,
            intor='cint3c2e_sph', label='j3c', max_memory=2000):
    ao_loc = cell.ao_loc_nr()
    aux_loc = auxcell.ao_loc_nr('ssc' in intor)
    nao = ao_loc[-1]
    naux = aux_loc[-1]
    nkptij = len(kptij_lst)

    xyz = numpy.asarray(cell.atom_coords(), order='C')
    ptr_coordL = cell._atm[:,PTR_COORD]
    ptr_coordL = numpy.vstack((ptr_coordL,ptr_coordL+1,ptr_coordL+2)).T.copy('C')

    kpti = kptij_lst[:,0]
    kptj = kptij_lst[:,1]
    aosym_s2 = numpy.einsum('ix->i', abs(kpti-kptj)) < 1e-9
    Ls = cell.get_lattice_Ls(cell.nimgs)

    if h5py.is_hdf5(erifile):
        feri = h5py.File(erifile)
    else:
        feri = h5py.File(erifile, 'w')

    workers = numpy.zeros(len(all_jobs), dtype=int)
    costs = [(ao_loc[i1]-ao_loc[i0]) for job_id, i0, i1 in all_jobs]
    for job_id, ish0, ish1 in mpi.work_balanced_partition(all_jobs, costs):
        dataname = '%s/%d' % (label, job_id)
        if dataname in feri:
            del(feri[dataname])

        di = ao_loc[ish1] - ao_loc[ish0]
        dij = di * nao
        buflen = max(8, int(max_memory*1e6/16/(nkptij*dij)))
        auxranges = balance_segs(aux_loc[1:]-aux_loc[:-1], buflen)
        buflen = max([x[2] for x in auxranges])
        buf = [numpy.zeros(dij*buflen, dtype=numpy.complex128) for k in range(nkptij)]
        ints = incore._wrap_int3c(cell, auxcell, intor, 1, Ls, buf)
        atm, bas, env = ints._envs[:3]

        for kpt_id, kptij in enumerate(kptij_lst):
            key = '%s/%d' % (dataname, kpt_id)
            shape = (naux, dij)
            if gamma_point(kptij):
                feri.create_dataset(key, shape, 'f8')
            else:
                feri.create_dataset(key, shape, 'c16')

        naux0 = 0
        for istep, auxrange in enumerate(auxranges):
            logger.debug1(cell, "job_id %d step %d", job_id, istep)
            sh0, sh1, nrow = auxrange
            c_shls_slice = (ctypes.c_int*6)(ish0, ish1, cell.nbas, cell.nbas*2,
                                            cell.nbas*2+sh0,
                                            cell.nbas*2+sh1)
            if numpy.all(aosym_s2):
                for l, L1 in enumerate(Ls):
                    env[ptr_coordL] = xyz + L1
                    e = numpy.dot(Ls[:l+1]-L1, kptj.T)  # Lattice sum over half of the images {1..l}
                    exp_Lk = numpy.exp(1j * numpy.asarray(e, order='C'))
                    exp_Lk[l] = .5
                    ints(exp_Lk, c_shls_slice)
            else:
                for l, L1 in enumerate(Ls):
                    env[ptr_coordL] = xyz + L1
                    e = numpy.dot(Ls, kptj.T) - numpy.dot(L1, kpti.T)
                    exp_Lk = numpy.exp(1j * numpy.asarray(e, order='C'))
                    ints(exp_Lk, c_shls_slice)

            for k, kptij in enumerate(kptij_lst):
                h5dat = feri['%s/%d'%(dataname,k)]
                mat = numpy.ndarray((di,nao,nrow), order='F',
                                    dtype=numpy.complex128, buffer=buf[k])
                mat = mat.transpose(2,0,1)
                if gamma_point(kptij):
                    mat = mat.real
                h5dat[naux0:naux0+nrow] = mat.reshape(nrow,-1)
                mat[:] = 0
            naux0 += nrow
        workers[job_id] = rank
    feri.close()
    workers = mpi.allreduce(workers)
    return workers

def is_zero(kpt):
    return abs(kpt).sum() < mdf.KPT_DIFF_TOL
gamma_point = is_zero

def async_write(thread_io, fn, *args):
    if thread_io is not None:
        thread_io.join()
    thread_io = lib.background_thread(fn, *args)
    return thread_io

if __name__ == '__main__':
    from pyscf.pbc import gto as pgto
    from mpi4pyscf.pbc import df
    cell = pgto.M(atom='He 0 0 0; He 0 0 1', h=numpy.eye(3)*4, gs=[5]*3)
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

