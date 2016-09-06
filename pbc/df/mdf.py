#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Exact density fitting with Gaussian and planewaves
Ref:
'''

import time
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
    if rank == 0:
# Note j2c may break symmetry
        j2c = gto.intor_cross('cint2c2e_sph', auxcell, nuccell)
        jaux = j2c.dot(charge)
    else:
        jaux = 0
    t1 = log.timer_debug1('vnuc pass1: analytic int', *t1)

    kpt_allow = numpy.zeros(3)
    coulG = tools.get_coulG(cell, kpt_allow, gs=mydf.gs) / cell.vol
    Gv = cell.get_Gv(mydf.gs)
    aoaux = ft_ao.ft_ao(nuccell, Gv)
    vGR = numpy.einsum('i,xi->x', charge, aoaux.real) * coulG
    vGI = numpy.einsum('i,xi->x', charge, aoaux.imag) * coulG

    max_memory = mydf.max_memory - lib.current_memory()[0]
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

    nao = cell.nao_nr()
    nao_pair = nao * (nao+1) // 2
    jaux -= charge.sum() * mydf.auxbar(auxcell)
    row_segs = list(lib.prange(0, jaux.size, mydf.blockdim))
    for k, kpt in enumerate(kpts_lst):
        with mydf.load_Lpq((kpt,kpt)) as Lpq:
            v = 0
            i1 = 0
            for p0, p1 in mpi.static_partition(row_segs):
                i0, i1 = i1, i1+p1-p0
                v += numpy.dot(jaux[p0:p1], numpy.asarray(Lpq[i0:i1]))
        if i1 > 0:
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

        # Use DF object to mimic KRHF/KUHF object in function get_coulG
        self.exxdiv = exxdiv

        if kpts.shape == (3,):
            return mdf_jk.get_jk(self, dm, hermi, kpts, kpt_band, with_j, with_k)

        vj = vk = None
        if with_k:
            vk = mdf_jk.get_k_kpts(self, dm, hermi, kpts, kpt_band)
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
    nao = cell.nao_nr()
    aux_loc = auxcell.ao_loc_nr()
    naux = aux_loc[-1]
    naux_short = auxcell_short.nao_nr()
    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    kpt_ji = kptjs - kptis
    uniq_kpts, uniq_index, uniq_inverse = mdf.unique(kpt_ji)

# Estimates the buffer size based on the last contraction in G-space.
# This contraction requires to hold nkptj copies of (naux,?) array
# simultaneously in memory.
    max_memory = mydf.max_memory - lib.current_memory()[0]
    max_memory = max(2000, min(comm.allgather(max_memory)))
    nkptj_max = max(numpy.unique(uniq_inverse, return_counts=True)[1])
    buflen = min(max(int(max_memory*.6*1e6/16/naux/(nkptj_max+1)), 1), nao**2)
    chunks = (min(int(max_memory*.6*1e6/buflen), naux), buflen)

    Lpq_jobs = grids2d_int3c_jobs(cell, auxcell_short, kptij_lst, chunks)
    j3c_jobs = grids2d_int3c_jobs(cell, auxcell, kptij_lst, chunks)
    fusion_jobs = grids2d_fusion_jobs(cell, auxcell, kptij_lst, chunks)

    if mydf.metric.upper() == 'S':
        Lpq_workers = _aux_e2(cell, auxcell_short, mydf._cderi, kptij_lst,
                              lib.flatten(Lpq_jobs),
                              'cint3c1e_sph', 'Lpq-chunks', max_memory)
        s_aux = auxcell_short.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=uniq_kpts)
    else:
        Lpq_workers = _aux_e2(cell, auxcell_short, mydf._cderi, kptij_lst,
                              lib.flatten(Lpq_jobs),
                              'cint3c1e_p2_sph', 'Lpq-chunks', max_memory)
        s_aux = auxcell_short.pbc_intor('cint1e_kin_sph', hermi=1, kpts=uniq_kpts)
        s_aux = [x*2 for x in s_aux]
    s_aux = [scipy.linalg.cho_factor(x) for x in s_aux]
    compress = mdf.compress_Lpq_to_chgcell(auxcell_short, chgcell)
    t1 = log.timer_debug1('Lpq', *t1)

    j3c_workers = _aux_e2(cell, auxcell, mydf._cderi, kptij_lst,
                          lib.flatten(j3c_jobs),
                          'cint3c2e_sph', 'j3c-chunks', max_memory)
    t1 = log.timer_debug1('3c2e', *t1)
    ####

    gs = mydf.gs
    gxyz = lib.cartesian_prod((numpy.append(range(gs[0]+1), range(-gs[0],0)),
                               numpy.append(range(gs[1]+1), range(-gs[1],0)),
                               numpy.append(range(gs[2]+1), range(-gs[2],0))))
    invh = numpy.linalg.inv(cell._h)
    Gv = 2*numpy.pi * numpy.dot(gxyz, invh)
    ngs = gxyz.shape[0]
    # j2c ~ (-kpt_ji | kpt_ji)
    j2c = auxcell.pbc_intor('cint2c2e_sph', hermi=1, kpts=uniq_kpts)
    kLRs = []
    kLIs = []
    for k, kpt in enumerate(uniq_kpts):
        aoaux = ft_ao.ft_ao(auxcell, Gv, None, invh, gxyz, gs, kpt)
        coulG = tools.get_coulG(cell, kpt, gs=gs) / cell.vol
        aoauxG = aoaux * coulG.reshape(-1,1)
        kLRs.append(numpy.asarray(aoauxG.real, order='C'))
        kLIs.append(numpy.asarray(aoauxG.imag, order='C'))

        if is_zero(kpt):  # kpti == kptj
            j2c[k] -= lib.dot(kLRs[-1].T, numpy.asarray(aoaux.real,order='C'))
            j2c[k] -= lib.dot(kLIs[-1].T, numpy.asarray(aoaux.imag,order='C'))
        else:
             # aoaux ~ kpt_ij, aoaux.conj() ~ kpt_kl
            j2c[k] -= lib.dot(aoauxG.conj().T, aoaux)

    feri = h5py.File(mydf._cderi)
    aosym_s2 = numpy.einsum('ix->i', abs(kptis-kptjs)) < 1e-9
    j_only = numpy.all(aosym_s2)
    fusion_workers = mpi.tasks_location(mpi.static_partition)(lib.flatten(fusion_jobs))
    def loop_fusion_jobs(uniq_k_id):
        for job in fusion_jobs[uniq_k_id]:
            job_id = job[0]
            worker = fusion_workers[job_id]
            col0, col1 = job[4:6]
            yield job_id, worker, col0, col1

    def collect_chunks(keys, workers, dst):
        val = []
        for i, worker in enumerate(workers):
            if rank == dst:
                if worker == dst:
                    val.append(numpy.asarray(feri[keys[i]]))
                else:
                    val.append(mpi.sendrecv(None, worker, dst))
            elif rank == worker:
                mpi.sendrecv(numpy.asarray(feri[keys[i]]), worker, dst)
        return val

    def distribute_int3c_s1(int3c_jobs, int3c_workers, col_id, label,
                            worker_dst, k):
        job_ids = [jobs[col_id][0] for jobs in int3c_jobs]
        keys = ['%s-chunks/%d/%d' % (label, job_id, k) for job_id in job_ids]
        workers = [int3c_workers[job_id] for job_id in job_ids]
        v = collect_chunks(keys, workers, worker_dst)
        if rank == worker_dst:
            return numpy.vstack(v)

    def distribute_int3c_s2(int3c_jobs, int3c_workers, label, k):
        uniq_k_id = uniq_inverse[k]
        for row_id, jobs in enumerate(int3c_jobs):
            job_ids = [job[0] for job in jobs]
            keys = ['%s-chunks/%d/%d' % (label, job_id, k) for job_id in job_ids]
            workers = [int3c_workers[job_id] for job_id in job_ids]
            v = collect_chunks(keys, workers, 0)

            if rank == 0:
                v = numpy.hstack(v).reshape(-1,nao,nao)
                if j_only:
                    tmp = numpy.asarray(v.transpose(0,2,1).conj(), order='C')
                    v += tmp
                    tmp = None
                v = lib.pack_tril(v)

            ksh0, ksh1 = jobs[0][1:3]
            row0, row1 = aux_loc[ksh0], aux_loc[ksh1]
            for job_id, worker, col0, col1 in loop_fusion_jobs(uniq_k_id):
                key = '%s-fus/%d/%d' % (label, job_id, k)
                if rank == 0:
                    if worker == 0:
                        feri[key][row0:row1] = v[:,col0:col1]
                    else:
                        mpi.sendrecv(v[:,col0:col1], 0, worker)
                elif rank == worker:
                    feri[key][row0:row1] = mpi.sendrecv(None, 0, worker)
    ####
    for k, kptij in enumerate(kptij_lst):
        uniq_k_id = uniq_inverse[k]

        if aosym_s2[k]:
            if gamma_point(kptij):
                dtype = 'f8'
            else:
                dtype = 'c16'
            for job_id, worker, col0, col1 in loop_fusion_jobs(uniq_k_id):
                ncol = col1 - col0
                if rank == worker:
                    key = '-fus/%d/%d' % (job_id, k)
                    feri.create_dataset('Lpq'+key, (naux,ncol), dtype)
                    feri.create_dataset('j3c'+key, (naux,ncol), dtype)

            distribute_int3c_s2(Lpq_jobs, Lpq_workers, 'Lpq', k)
            distribute_int3c_s2(j3c_jobs, j3c_workers, 'j3c', k)

            for job_id, worker, col0, col1 in loop_fusion_jobs(uniq_k_id):
                if rank == worker:
                    key = '-fus/%d/%d' % (job_id, k)
                    Lpq = feri['Lpq'+key][:naux_short]
                    Lpq = compress(scipy.linalg.cho_solve(s_aux[uniq_k_id], Lpq))
                    feri['Lpq'+key][:] = Lpq
                    j3c = numpy.asarray(feri['j3c'+key])
                    lib.dot(j2c[uniq_k_id], Lpq, -.5, j3c, 1)
                    feri['j3c'+key][:] = j3c
                Lpq = j3c = None

        else:
            for col_id, (job_id, worker, col0, col1) \
                    in enumerate(loop_fusion_jobs(uniq_k_id)):
                Lpq = distribute_int3c_s1(Lpq_jobs, Lpq_workers, col_id, 'Lpq', worker, k)
                j3c = distribute_int3c_s1(j3c_jobs, j3c_workers, col_id, 'j3c', worker, k)

                if rank == worker:
                    key = '-fus/%d/%d' % (job_id, k)
                    Lpq = compress(scipy.linalg.cho_solve(s_aux[uniq_k_id], Lpq))
                    feri['Lpq'+key] = Lpq
                    lib.dot(j2c[uniq_k_id], Lpq, -.5, j3c, 1)
                    feri['j3c'+key] = j3c
                Lpq = j3c = None
    aoaux = coulG = s_aux = j2c = None
    if 'Lpq-chunks' in feri: del(feri['Lpq-chunks'])
    if 'j3c-chunks' in feri: del(feri['j3c-chunks'])
    comm.Barrier()
    t1 = log.timer_debug1('distributing Lpq j3c', *t1)
    ####

    vbar = mydf.auxbar(auxcell)
    ovlp = cell.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=kptjs[aosym_s2])
    ovlp = [lib.pack_tril(x) for x in ovlp]

    def process(job_id, uniq_kptji_id, sh0, sh1, col0, col1):
        kpt = uniq_kpts[uniq_kptji_id]  # kpt = kptj - kpti
        adapted_ji_idx = numpy.where(uniq_inverse == uniq_kptji_id)[0]
        adapted_kptjs = kptjs[adapted_ji_idx]
        nkptj = len(adapted_kptjs)
        kLR = kLRs[uniq_kptji_id]
        kLI = kLIs[uniq_kptji_id]

        j3cR = []
        j3cI = []
        for idx in adapted_ji_idx:
            v = numpy.asarray(feri['j3c-fus/%d/%d'%(job_id,idx)])
            j3cR.append(numpy.asarray(v.real, order='C'))
            if v.dtype == numpy.complex128:
                j3cI.append(numpy.asarray(v.imag, order='C'))
            else:
                j3cI.append(None)

        ncol = col1 - col0
        if is_zero(kpt):  # kpti == kptj
            aosym = 's2'
            Gblksize = max(16, int(max_memory*.2*1e6/16/ncol/(nkptj+1)))  # +1 for pqkRbuf/pqkIbuf
        else:
            aosym = 's1'
            Gblksize = max(16, int(max_memory*.4*1e6/16/ncol/(nkptj+1)))
        Gblksize = min(Gblksize, ngs)
        pqkRbuf = numpy.empty(ncol*Gblksize)
        pqkIbuf = numpy.empty(ncol*Gblksize)
        # buf for ft_aopair
        buf = numpy.empty((nkptj,buflen*Gblksize), dtype=numpy.complex128)

        if is_zero(kpt):  # kpti == kptj
            shls_slice = (sh0, sh1, 0, sh1)
            for p0, p1 in lib.prange(0, ngs, Gblksize):
                ft_ao._ft_aopair_kpts(cell, Gv[p0:p1], shls_slice, aosym, invh,
                                      gxyz[p0:p1], gs, kpt, adapted_kptjs, out=buf)
                nG = p1 - p0
                for k, ji in enumerate(adapted_ji_idx):
                    aoao = numpy.ndarray((nG,ncol), dtype=numpy.complex128,
                                         order='F', buffer=buf[k])
                    pqkR = numpy.ndarray((ncol,nG), buffer=pqkRbuf)
                    pqkI = numpy.ndarray((ncol,nG), buffer=pqkIbuf)
                    pqkR[:] = aoao.real.T
                    pqkI[:] = aoao.imag.T
                    aoao[:] = 0
                    lib.dot(kLR[p0:p1].T, pqkR.T, -1, j3cR[k], 1)
                    lib.dot(kLI[p0:p1].T, pqkI.T, -1, j3cR[k], 1)
                    if not (is_zero(kpt) and gamma_point(adapted_kptjs[k])):
                        lib.dot(kLR[p0:p1].T, pqkI.T, -1, j3cI[k], 1)
                        lib.dot(kLI[p0:p1].T, pqkR.T,  1, j3cI[k], 1)
        else:
            shls_slice = (sh0, sh1, 0, cell.nbas)
            ni = ncol // nao
            for p0, p1 in lib.prange(0, ngs, Gblksize):
                ft_ao._ft_aopair_kpts(cell, Gv[p0:p1], shls_slice, aosym, invh,
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
                    zdotCN(kLR[p0:p1].T, kLI[p0:p1].T, pqkR.T, pqkI.T,
                           -1, j3cR[k], j3cI[k], 1)

        if is_zero(kpt):
            for k, ji in enumerate(adapted_ji_idx):
                if is_zero(adapted_kptjs[k]):
                    for i, c in enumerate(vbar):
                        if c != 0:
                            j3cR[k][i] -= c * ovlp[k][col0:col1].real
                else:
                    for i, c in enumerate(vbar):
                        if c != 0:
                            j3cR[k][i] -= c * ovlp[k][col0:col1].real
                            j3cI[k][i] -= c * ovlp[k][col0:col1].imag

        for k, idx in enumerate(adapted_ji_idx):
            if is_zero(kpt) and gamma_point(adapted_kptjs[k]):
                feri['j3c-fus/%d/%d'%(job_id,idx)][:] = j3cR[k]
            else:
                feri['j3c-fus/%d/%d'%(job_id,idx)][:] = j3cR[k] + j3cI[k]*1j
    ####
    for job in mpi.static_partition(lib.flatten(fusion_jobs)):
        process(*job)
    t1 = log.timer_debug1('fusing Lpq j3c', *t1)

    row_segs = list(lib.prange(0, naux, mydf.blockdim))
    nrow = sum([p1-p0 for p0, p1 in mpi.static_partition(row_segs)])
    if 'Lpq' in feri: del(feri['Lpq'])
    if 'j3c' in feri: del(feri['j3c'])
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

    def assemble(jobs, label, row0, row1, i0, k, worker_dst):
        if aosym_s2[k]:
            shape = (row1-row0, nao*(nao+1)//2)
        else:
            shape = (row1-row0, nao**2)
        if gamma_point(kptij_lst[k]):
            buf = numpy.empty(shape)
        else:
            buf = numpy.empty(shape, numpy.complex128)

        for job_id, uniq_k_id, sh0, sh1, col0, col1 in jobs:
            worker = fusion_workers[job_id]
            key = '%s-fus/%d/%d' % (label, job_id, k)
            if rank == worker_dst:
                if rank == worker:
                    buf[:,col0:col1] = feri[key][row0:row1]
                else:
                    buf[:,col0:col1] = mpi.sendrecv(None, worker, worker_dst)
            elif rank == worker:
                mpi.sendrecv(feri[key][row0:row1], worker, worker_dst)
        if rank == worker_dst:
            i1 = i0 + row1 - row0
            feri['%s/%d'%(label,k)][i0:i1] = buf

    j3c_workers = mpi.tasks_location(mpi.static_partition)(row_segs)
    for jobs in fusion_jobs:
        save_row0 = 0
        for i, (row0, row1) in enumerate(row_segs):
            worker_dst = j3c_workers[i]
            uniq_k_id = jobs[0][1]
            adapted_ji_idx = numpy.where(uniq_inverse == uniq_k_id)[0]

            for idx in adapted_ji_idx:
                assemble(jobs, 'Lpq', row0, row1, save_row0, idx, worker_dst)
                assemble(jobs, 'j3c', row0, row1, save_row0, idx, worker_dst)
            if worker_dst:
                save_row0 += row1 - row0

    if 'Lpq-fus' in feri: del(feri['Lpq-fus'])
    if 'j3c-fus' in feri: del(feri['j3c-fus'])
    t1 = log.timer_debug1('assembling Lpq j3c', *t1)

    if 'Lpq-kptij' in feri: del(feri['Lpq-kptij'])
    if 'Lpq-kptij' in feri: del(feri['j3c-kptij'])
    feri['Lpq-kptij'] = kptij_lst
    feri['j3c-kptij'] = kptij_lst
    feri.close()

def grids2d_int3c_jobs(cell, auxcell, kptij_lst, chunks):
    ao_loc = cell.ao_loc_nr()
    nao = ao_loc[-1]
    segs = (ao_loc[1:]-ao_loc[:-1])*nao
    ij_ranges = balance_segs(segs, chunks[1])

    ao_loc = auxcell.ao_loc_nr()
    segs = ao_loc[1:] - ao_loc[:-1]
    k_ranges = balance_segs(segs, chunks[0])

    jobs = []
    job_id = 0
    for k0, k1, krow in k_ranges:
        jobs_by_row = []
        for i0, i1, icol in ij_ranges:
            jobs_by_row.append((job_id, k0, k1, i0, i1))
            job_id += 1
        jobs.append(jobs_by_row)
    return jobs

def grids2d_fusion_jobs(cell, auxcell, kptij_lst, chunks):
    kptis = kptij_lst[:,0]
    kptjs = kptij_lst[:,1]
    kpt_ji = kptjs - kptis
    uniq_kpts, uniq_index, uniq_inverse = mdf.unique(kpt_ji)
    ao_loc = cell.ao_loc_nr()
    nao = ao_loc[-1]

    segs = (ao_loc[1:]-ao_loc[:-1])*nao
    ij_ranges_s1 = balance_segs(segs, chunks[1])
    segs = ao_loc[1:]*(ao_loc[1:]+1)//2 - ao_loc[:-1]*(ao_loc[:-1]+1)//2
    ij_ranges_s2 = balance_segs(segs, chunks[1])

    jobs = []
    job_id = 0
    for k, kpt in enumerate(uniq_kpts):
        if is_zero(kpt):  # kpti == kptj
            shranges = ij_ranges_s2
        else:
            shranges = ij_ranges_s1
        jobs_for_k = []
        col1 = 0
        for sh_range in shranges:
            sh0, sh1, ncol = sh_range
            col0, col1 = col1, col1+ncol
            jobs_for_k.append((job_id, k, sh0, sh1, col0, col1))
            job_id += 1
        jobs.append(jobs_for_k)
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

    nao = cell.nao_nr()
    #:buf = [numpy.zeros((nao,nao), order='F', dtype=numpy.complex128)
    #:       for k in range(nkpts)]
    buf = numpy.zeros((nkpts,fakenuc.natm,nao,nao),
                      dtype=numpy.complex128).transpose(0,3,2,1)
    ints = incore._wrap_int3c(cell, fakenuc, 'cint3c2e_sph', 1, Ls, buf)
    atm, bas, env = ints._envs[:3]
    c_shls_slice = (ctypes.c_int*6)(0, cell.nbas, cell.nbas, cell.nbas*2,
                                    cell.nbas*2, cell.nbas*2+fakenuc.natm)

    xyz = numpy.asarray(cell.atom_coords(), order='C')
    ptr_coordL = atm[:cell.natm,PTR_COORD]
    ptr_coordL = numpy.vstack((ptr_coordL,ptr_coordL+1,ptr_coordL+2)).T.copy('C')
    for l in mpi.static_partition(range(len(Ls))):
        L1 = Ls[l]
        env[ptr_coordL] = xyz + L1
        exp_Lk = numpy.einsum('k,ik->ik', expLk[l].conj(), expLk[:l+1])
        exp_Lk = numpy.asarray(exp_Lk, order='C')
        exp_Lk[l] = .5
        ints(exp_Lk, c_shls_slice)

    charge = cell.atom_charges()
    charge = numpy.append(charge, -charge)  # (charge-of-nuccell, charge-of-fakenuc)
    buf = numpy.einsum('kijz,z->kij', buf, charge)
    buf = buf + buf.transpose(0,2,1).conj()
# buf is mpi.reduced in get_nuc function
    return buf

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
    costs = [(aux_loc[k1]-aux_loc[k0])*(aux_loc[i1]-aux_loc[i0])
             for job_id, k0, k1, i0, i1 in all_jobs]
    for job_id, ksh0, ksh1, ish0, ish1 in mpi.work_balanced_partition(all_jobs, costs):
        dataname = '%s/%d' % (label, job_id)
        if dataname in feri:
            del(feri[dataname])

        di = ao_loc[ish1] - ao_loc[ish0]
        dk = aux_loc[ksh1] - aux_loc[ksh0]
        dij = di * nao
        buflen = max(8, int(max_memory*1e6/16/(nkptij*dij)))
        auxranges = balance_segs(aux_loc[ksh0+1:ksh1+1]-aux_loc[ksh0:ksh1], buflen)
        buflen = max([x[2] for x in auxranges])
        buf = [numpy.zeros(dij*buflen, dtype=numpy.complex128) for k in range(nkptij)]
        ints = incore._wrap_int3c(cell, auxcell, intor, 1, Ls, buf)
        atm, bas, env = ints._envs[:3]

        for kpt_id, kptij in enumerate(kptij_lst):
            key = '%s/%d' % (dataname, kpt_id)
            shape = (dk, dij)
            if gamma_point(kptij):
                feri.create_dataset(key, shape, 'f8')
            else:
                feri.create_dataset(key, shape, 'c16')

        naux0 = 0
        for istep, auxrange in enumerate(auxranges):
            #logger.debug1(cell, "job_id %d step %d", job_id, istep)
            sh0, sh1, nrow = auxrange
            c_shls_slice = (ctypes.c_int*6)(ish0, ish1, cell.nbas, cell.nbas*2,
                                            cell.nbas*2+ksh0+sh0,
                                            cell.nbas*2+ksh0+sh1)
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

