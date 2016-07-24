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
import h5py

from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc import tools
from pyscf.pbc.df import incore
from pyscf.pbc.df import outcore
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import xdf

from espy.lib import logger
from espy.tools import mpi
from espy.pbc.df import xdf_jk

comm = mpi.comm
rank = mpi.rank


def init_XDF(cell, kpts=numpy.zeros((1,3))):
    mydf = mpi.pool.apply(_init_XDF_wrap, [cell, kpts], [cell.dumps(), kpts])
    return mydf
def _init_XDF_wrap(args):
    from espy.pbc.df import xdf
    cell, kpts = args
    if xdf.rank > 0:
        cell = xdf.gto.loads(cell)
        cell.verbose = 0
    return xdf.mpi.register_for(xdf.XDF(cell, kpts))

def get_nuc(mydf, kpts=None):
    if mydf._cderi is None:
        mydf.build()
    args = (mydf._reg_keys, kpts)
    return mpi.pool.apply(_get_nuc_wrap, args, args)
def _get_nuc_wrap(args):
    from espy.pbc.df import xdf
    return xdf._get_nuc(*args)
def _get_nuc(reg_keys, kpts=None):
    mydf = xdf_jk._load_df(reg_keys)
    cell = mydf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = t0 = (time.clock(), time.time())
    nao = cell.nao_nr()
    auxcell = mydf.auxcell
    nuccell = xdf.make_modchg_basis(cell, mydf.eta, 0)
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
        jaux  = j2c.dot(charge)
        jaux -= charge.sum() * mydf.auxbar(auxcell)
    else:
        jaux = 0
    t1 = log.timer_debug1('vnuc pass1: analytic int', *t1)

    kpt_allow = numpy.zeros(3)
    coulG = tools.get_coulG(cell, kpt_allow, gs=mydf.gs) / cell.vol

# Append nuccell to auxcell, so that they can be FT together in pw_loop
# the first [:naux] of ft_ao are aux fitting functions.
    nuccell._atm, nuccell._bas, nuccell._env = \
            gto.conc_env(auxcell._atm, auxcell._bas, auxcell._env,
                         nuccell._atm, nuccell._bas, nuccell._env)
    naux = auxcell.nao_nr()

    max_memory = mydf.max_memory - lib.current_memory()[0]
    for k, pqkR, LkR, pqkI, LkI, p0, p1 \
            in mydf.mpi_ft_loop(cell, nuccell, mydf.gs, kpt_allow, kpts_lst, max_memory):
# rho_ij(G) nuc(-G) / G^2
# = [Re(rho_ij(G)) + Im(rho_ij(G))*1j] [Re(nuc(G)) - Im(nuc(G))*1j] / G^2
        vGR = numpy.einsum('i,ix->x', charge, LkR[naux:]) * coulG[p0:p1]
        vGI = numpy.einsum('i,ix->x', charge, LkI[naux:]) * coulG[p0:p1]
        if abs(kpts_lst[k]).sum() > 1e-9:  # if not gamma point
            vj[k] += numpy.einsum('k,xk->x', vGR, pqkI) * 1j
            vj[k] += numpy.einsum('k,xk->x', vGI, pqkR) *-1j
        vj[k] += numpy.einsum('k,xk->x', vGR, pqkR)
        vj[k] += numpy.einsum('k,xk->x', vGI, pqkI)
        if k == 0:
            jaux -= numpy.einsum('k,xk->x', vGR, LkR[:naux])
            jaux -= numpy.einsum('k,xk->x', vGI, LkI[:naux])
    t1 = log.timer_debug1('contracting Vnuc', *t1)
    vj = mpi.reduce(lib.asarray(vj))
    jaux = mpi.reduce(jaux)

    if rank == 0:
        ovlp = cell.pbc_intor('cint1e_ovlp_sph', 1, lib.HERMITIAN, kpts_lst)
        nao_pair = nao * (nao+1) // 2
        vj = vj.reshape(-1,nao,nao)
        for k, kpt in enumerate(kpts_lst):
            vj[k] -= nucbar * ovlp[k]
            for Lpq in mydf.load_Lpq((kpt,kpt)):
                vpq = numpy.dot(jaux, Lpq)
                if vpq.shape == nao_pair:
                    vpq = lib.unpack_tril(vpq)
                vj[k] += vpq

        if abs(kpts_lst).sum() < 1e-9:  # gamma point
            vj = vj.real
        if kpts is None or numpy.shape(kpts) == (3,):
            vj = vj[0]
        return vj


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


class XDF(xdf.XDF):
    def __enter__(self):
        return self
    def __exit__(self):
        self.close()
    def close(self):
        self._reg_keys = mpi.del_registry(self._reg_keys)

    def build(self, j_only=False, with_Lpq=True, with_j3c=True):
        args = (self._reg_keys, j_only, with_Lpq, with_j3c)
        mpi.pool.apply(_build_wrap, args, args)
        return self

    def mpi_pw_loop(self, cell, auxcell, gs=None, kpti_kptj=None, max_memory=2000):
        if gs is None:
            gs = self.gs
        if kpti_kptj is None:
            kpti = kptj = numpy.zeros(3)
        else:
            kpti, kptj = kpti_kptj

        nao = cell.nao_nr()
        naux = auxcell.nao_nr()
        gxrange = numpy.append(range(gs[0]+1), range(-gs[0],0))
        gyrange = numpy.append(range(gs[1]+1), range(-gs[1],0))
        gzrange = numpy.append(range(gs[2]+1), range(-gs[2],0))
        gxyz = lib.cartesian_prod((gxrange, gyrange, gzrange))
        invh = numpy.linalg.inv(cell._h)
        Gv = 2*numpy.pi * numpy.dot(gxyz, invh)
        ngs = gxyz.shape[0]

# Theoretically, hermitian symmetry can be also found for kpti == kptj:
#       f_ji(G) = \int f_ji exp(-iGr) = \int f_ij^* exp(-iGr) = [f_ij(-G)]^*
# The hermi operation needs reordering the axis-0.  It is inefficient
        hermi = abs(kpti).sum() < 1e-9 and abs(kptj).sum() < 1e-9  # gamma point

        blksize = min(max(16, int(max_memory*1e6*.7/16/nao**2)), 16384)
        sublk = max(16, int(blksize//4))
        pqkRbuf = numpy.empty(nao*nao*sublk)
        pqkIbuf = numpy.empty(nao*nao*sublk)
        LkRbuf = numpy.empty(naux*sublk)
        LkIbuf = numpy.empty(naux*sublk)

        for p0, p1 in mpi_prange(0, ngs, blksize):
            aoao = ft_ao.ft_aopair(cell, Gv[p0:p1], None, hermi, invh, gxyz[p0:p1],
                                   gs, (kpti, kptj))
            aoaux = ft_ao.ft_ao(auxcell, Gv[p0:p1], None, invh, gxyz[p0:p1],
                                gs, kptj-kpti)

            for i0, i1 in lib.prange(0, p1-p0, sublk):
                nG = i1 - i0
                pqkR = numpy.ndarray((nao,nao,nG), buffer=pqkRbuf)
                pqkI = numpy.ndarray((nao,nao,nG), buffer=pqkIbuf)
                pqkR[:] = aoao[i0:i1].real.transpose(1,2,0)
                pqkI[:] = aoao[i0:i1].imag.transpose(1,2,0)
                kLR = numpy.ndarray((nG,naux), buffer=LkRbuf)
                kLI = numpy.ndarray((nG,naux), buffer=LkIbuf)
                kLR [:] = aoaux[i0:i1].real
                kLI [:] = aoaux[i0:i1].imag
                yield (pqkR.reshape(-1,nG), kLR.T,
                       pqkI.reshape(-1,nG), kLI.T, p0+i0, p0+i1)

    def mpi_ft_loop(self, cell, auxcell, gs=None, kpt=numpy.zeros(3),
                    kpts=None, max_memory=4000):
        if gs is None: gs = self.gs
        if kpts is None:
            assert(abs(kpt).sum() < 1e-9)
            kpts = self.kpts
        kpts = numpy.asarray(kpts)
        nkpts = len(kpts)

        nao = cell.nao_nr()
        naux = auxcell.nao_nr()
        gxrange = numpy.append(range(gs[0]+1), range(-gs[0],0))
        gyrange = numpy.append(range(gs[1]+1), range(-gs[1],0))
        gzrange = numpy.append(range(gs[2]+1), range(-gs[2],0))
        gxyz = lib.cartesian_prod((gxrange, gyrange, gzrange))
        invh = numpy.linalg.inv(cell._h)
        Gv = 2*numpy.pi * numpy.dot(gxyz, invh)
        ngs = gxyz.shape[0]

        blksize = min(max(16, int(max_memory*1e6*.9/(nao**2*(nkpts+1)*16))), 16384)
        buf = [numpy.zeros(nao*nao*blksize, dtype=numpy.complex128)
               for k in range(nkpts)]
        pqkRbuf = numpy.empty(nao*nao*blksize)
        pqkIbuf = numpy.empty(nao*nao*blksize)
        LkRbuf = numpy.empty(naux*blksize)
        LkIbuf = numpy.empty(naux*blksize)

        for p0, p1 in mpi_prange(0, ngs, blksize):
            aoaux = ft_ao.ft_ao(auxcell, Gv[p0:p1], None, invh,
                                gxyz[p0:p1], gs, kpt)
            nG = p1 - p0
            LkR = numpy.ndarray((naux,nG), buffer=LkRbuf)
            LkI = numpy.ndarray((naux,nG), buffer=LkIbuf)
            LkR [:] = aoaux.real.T
            LkI [:] = aoaux.imag.T

            ft_ao._ft_aopair_kpts(cell, Gv[p0:p1], None, True, invh,
                                  gxyz[p0:p1], gs, kpt, kpts, out=buf)
            for k in range(nkpts):
                aoao = numpy.ndarray((nG,nao,nao), dtype=numpy.complex128,
                                     order='F', buffer=buf[k])
                pqkR = numpy.ndarray((nao,nao,nG), buffer=pqkRbuf)
                pqkI = numpy.ndarray((nao,nao,nG), buffer=pqkIbuf)
                pqkR[:] = aoao.real.transpose(1,2,0)
                pqkI[:] = aoao.imag.transpose(1,2,0)
                yield (k, pqkR.reshape(-1,nG), LkR, pqkI.reshape(-1,nG), LkI, p0, p1)
                aoao[:] = 0

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
            return xdf_jk.get_jk(self, dm, hermi, kpts, kpt_band, with_j, with_k)

        vj = vk = None
        if with_k:
            vk = xdf_jk.get_k_kpts(self, dm, hermi, kpts, kpt_band)
        if with_j:
            vj = xdf_jk.get_j_kpts(self, dm, hermi, kpts, kpt_band)
        return vj, vk


def mpi_prange(start, stop, step):
    mpi_size = mpi.pool.size
    step = min(step, (stop-start+mpi_size-1)//mpi_size)
    task_lst = [(p0,p1) for p0, p1 in lib.prange(start, stop, step)]
    return mpi.static_partition(task_lst)


def _build_wrap(args):
    from espy.pbc.df import xdf
    return xdf._build(*args)
def _build(reg_keys, j_only=False, with_Lpq=True, with_j3c=True):
# Unlike DF and PWDF class, here XDF objects are synced once
    mydf = mpi._registry[reg_keys[rank]]
    mydf.kpts, mydf.gs, mydf.metric, mydf.approx_sr_level, mydf.auxbasis, \
            mydf.eta, mydf.exxdiv = \
            comm.bcast((mydf.kpts, mydf.gs, mydf.metric, mydf.approx_sr_level,
                        mydf.auxbasis, mydf.eta, mydf.exxdiv))

    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = (time.clock(), time.time())
    cell = mydf.cell
    if mydf.eta is None:
        mydf.eta = xdf.estimate_eta(cell)
        log.debug('Set smooth gaussian eta to %.9g', mydf.eta)
    mydf.dump_flags()

    auxcell = xdf.make_modrho_basis(cell, mydf.auxbasis, mydf.eta)
    chgcell = xdf.make_modchg_basis(auxcell, mydf.eta)

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
    if len(mydf.kpts) == 1:
        aosym = 's2ij'
    else:
        aosym = 's1'

    if with_Lpq:
        if mydf.approx_sr_level == 0:
            build_Lpq_pbc(mydf, auxcell, chgcell, aosym, kptij_lst)
        elif mydf.approx_sr_level == 1:
            build_Lpq_pbc(mydf, auxcell, chgcell, aosym, numpy.zeros((1,2,3)))
        elif mydf.approx_sr_level == 2:
            xdf.build_Lpq_nonpbc(mydf, auxcell, chgcell)
        elif mydf.approx_sr_level == 3:
            xdf.build_Lpq_1c_approx(mydf, auxcell, chgcell)
        elif mydf.approx_sr_level == 4:
            xdf.build_Lpq_atomic(mydf, auxcell, chgcell, mydf.eta)
    else:
        if mydf.approx_sr_level == 0:
            _distribute_int3c(mydf._cderi, 'Lpq', mydf.kpts, kptij_lst)
        else:
            with h5py.File(mydf._cderi) as feri:
                feri['Lpq-kptij'] = numpy.zeros((1,2,3))
                if rank == 0:
                    mpi.bcast(feri['Lpq'].value)
                else:
                    feri['Lpq'] = mpi.bcast(None)

# Merge chgcell into auxcell
    auxcell._atm, auxcell._bas, auxcell._env = \
            gto.conc_env(auxcell._atm, auxcell._bas, auxcell._env,
                         chgcell._atm, chgcell._bas, chgcell._env)
    mydf.auxcell = auxcell
    t1 = log.timer_debug1('Lpq', *t1)

    if with_j3c:
        build_j3c_pbc(mydf, auxcell, aosym, kptij_lst)
        t1 = log.timer_debug1('3c2e', *t1)
    else:
        _distribute_int3c(mydf._cderi, 'j3c', mydf.kpts, kptij_lst)
    return mydf

def build_j3c_pbc(mydf, auxcell, aosym, kptij_lst):
    if mpi.pool.size == 1:
        outcore.aux_e2(mydf.cell, auxcell, mydf._cderi, 'cint3c2e_sph',
                       aosym=aosym, kptij_lst=kptij_lst, dataname='j3c',
                       max_memory=mydf.max_memory)
        return

    aux_loc = auxcell.ao_loc_nr()
    dims = aux_loc[1:] - aux_loc[:-1]
    shl_lst = mpi.work_balanced_partition(numpy.arange(auxcell.nbas), dims)
    segcell = copy.copy(auxcell)
    segcell._bas = numpy.asarray(segcell._bas[shl_lst], order='C')

    outcore.aux_e2(mydf.cell, segcell, mydf._cderi, 'cint3c2e_sph',
                   aosym=aosym, kptij_lst=kptij_lst, dataname='j3c',
                   max_memory=mydf.max_memory)
    _assemble_int3c(mydf._cderi, 'j3c', mydf.kpts, kptij_lst)

def build_Lpq_pbc(mydf, auxcell, chgcell, aosym, kptij_lst):
    if mpi.pool.size == 1:
        return xdf.build_Lpq_pbc(mydf, auxcell, chgcell, aosym, kptij_lst)

    # Each worker processor computes a segment of the auxbasis
    aux_loc = auxcell.ao_loc_nr()
    dims = aux_loc[1:] - aux_loc[:-1]
    shl_lst = mpi.work_balanced_partition(numpy.arange(auxcell.nbas), dims)
    segcell = copy.copy(auxcell)
    segcell._bas = numpy.asarray(segcell._bas[shl_lst], order='C')

    kpts_ji = kptij_lst[:,1] - kptij_lst[:,0]
    if mydf.metric.upper() == 'S':
        outcore.aux_e2(mydf.cell, segcell, mydf._cderi, 'cint3c1e_sph',
                       aosym=aosym, kptij_lst=kptij_lst, dataname='Lpq',
                       max_memory=mydf.max_memory)
        j2c = auxcell.pbc_intor('cint1e_ovlp_sph', hermi=1, kpts=kpts_ji)
    else:  # mydf.metric.upper() == 'T'
        outcore.aux_e2(mydf.cell, segcell, mydf._cderi, 'cint3c1e_p2_sph',
                       aosym=aosym, kptij_lst=kptij_lst, dataname='Lpq',
                       max_memory=mydf.max_memory)
        j2c = [x*2 for x in auxcell.pbc_intor('cint1e_kin_sph', hermi=1, kpts=kpts_ji)]

    _assemble_int3c(mydf._cderi, 'Lpq', mydf.kpts, kptij_lst)

    with h5py.File(mydf._cderi) as feri:
        for k, j2c_k in enumerate(j2c):
            key = 'Lpq/%d' % k
            if key in feri:
                Lpq = feri[key].value
                del(feri[key])
                Lpq = lib.cho_solve(j2c_k, Lpq)
                feri[key] = xdf.compress_Lpq_to_chgcell(Lpq, auxcell, chgcell)


# Note on each proccessor, _int_nuc_vloc computes only a fraction of the entire vj.
# It is because the summation over real space images are splited by mpi.static_partition
def _int_nuc_vloc(cell, nuccell, kpts):
    '''Vnuc - Vloc'''
    nimgs = numpy.max((cell.nimgs, nuccell.nimgs), axis=0)
    Ls = numpy.asarray(cell.get_lattice_Ls(nimgs), order='C')
    expLk = numpy.asarray(numpy.exp(1j*numpy.dot(Ls, kpts.T)), order='C')
    nkpts = len(kpts)

    fakenuc = xdf._fake_nuc(cell)
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
    ptr_coordL = atm[:cell.natm,gto.mole.PTR_COORD]
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
    return buf

def _unique_kpt_ji(kpts):
    nkpts = len(kpts)
    kk_table = kpts.reshape(-1,1,3) - kpts.reshape(1,-1,3)
    todo = numpy.ones((nkpts,nkpts), dtype=bool)
    uniq_kptji = []
    for kj, kptj in enumerate(kpts):
        for ki, kpti in enumerate(kpts):
            if todo[ki,kj]:
                kpt = kptj - kpti
                uniq_kptji.append(kpt)
                kk_match = numpy.einsum('ijx->ij', abs(kk_table + kpt)) < 1e-9
                todo[kk_match  ] = False
                todo[kk_match.T] = False
    return numpy.array(uniq_kptji)

def _assign_kpts_task(kpts, kptij_lst):
    mask = numpy.zeros(len(kptij_lst), dtype=bool)
    uniq_kptji = _unique_kpt_ji(kpts)
    kpti_lst = kptij_lst[:,0]
    kptj_lst = kptij_lst[:,1]
    kpts_ji = kptj_lst - kpti_lst
    for kpt in mpi.static_partition(uniq_kptji):
        mask |= numpy.einsum('ki->k', abs(kpts_ji-kpt)) < 1e-9

    worker = numpy.zeros(len(kptij_lst), dtype=int)
    worker[mask] = rank
    worker = mpi.allreduce(worker)
    return worker

def _assemble_int3c(ints_file, dataname, kpts, kptij_lst):
    kptij_worker = _assign_kpts_task(kpts, kptij_lst)
    with h5py.File(ints_file) as feri:
        for k, proc_id in enumerate(kptij_worker):
            key = '%s/%d' % (dataname,k)
            dat = mpi.gather(feri[key].value)
            dat = mpi.sendrecv(dat, 0, proc_id)
            del(feri[key])
            if rank == 0 or proc_id == rank:
                feri[key] = dat

def _distribute_int3c(ints_file, dataname, kpts, kptij_lst):
    kptij_worker = _assign_kpts_task(kpts, kptij_lst)
    with h5py.File(ints_file) as feri:
        if rank == 0:
            kptij_lst = feri['%s-kptij'%dataname].value
            mpi.bcast(kptij_lst)
        else:
            kptij_lst = mpi.bcast(None)
            feri['%s-kptij'%dataname] = kptij_lst

        for k, proc_id in enumerate(kptij_worker):
            if proc_id != 0:
                key = '%s/%d' % (dataname,k)
                if rank == 0:
                    dat = feri[key].value
                else:
                    dat = None
                dat = mpi.sendrecv(dat, 0, proc_id)
                if rank == proc_id:
                    if key in feri:
                        del(feri[key])
                    feri[key] = dat

if __name__ == '__main__':
    from pyscf.pbc import gto as pgto
    from espy.pbc import df
    cell = pgto.M(atom='He 0 0 0; He 0 0 1', h=numpy.eye(3)*4, gs=[5]*3)
    mydf = df.XDF(cell, kpts)

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

