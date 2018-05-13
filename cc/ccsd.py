import time
import ctypes
from functools import reduce
import numpy
from pyscf import gto
from pyscf import lib
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf.cc import ccsd
from pyscf import __config__

from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi

BLKMIN = getattr(__config__, 'cc_ccsd_blkmin', 4)
MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

comm = mpi.comm
rank = mpi.rank


# t1: ia
# t2: ijab
def kernel(mycc, eris, t1=None, t2=None, max_cycle=50, tol=1e-8, tolnormt=1e-6,
           verbose=None):
    log = logger.new_logger(mycc, verbose)

    if t1 is None and t2 is None:
        t1, t2 = mycc.get_init_guess(eris)
    elif t2 is None:
        t2 = mycc.get_init_guess(eris)[1]

    cput1 = cput0 = (time.clock(), time.time())
    eold = 0
    vec_old = 0
    eccsd = 0
    if isinstance(mycc.diis, lib.diis.DIIS):
        adiis = mycc.diis
    elif mycc.diis:
        adiis = lib.diis.DIIS(mycc, mycc.diis_file, incore=mycc.incore_complete)
        adiis.space = mycc.diis_space
    else:
        adiis = None

    conv = False
    for istep in range(max_cycle):
        t1new, t2new = mycc.update_amps(t1, t2, eris)
        normt = numpy.linalg.norm(mycc.amplitudes_to_vector(t1new, t2new) -
                                  mycc.amplitudes_to_vector(t1, t2))
        t1, t2 = t1new, t2new
        t1new = t2new = None
# mpi_diis
        t1, t2 = mycc.run_diis(t1, t2, istep, normt, eccsd-eold, adiis)
        eold, eccsd = eccsd, mycc.energy(t1, t2, eris)
        log.info('cycle = %d  E(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                 istep+1, eccsd, eccsd - eold, normt)
        cput1 = log.timer('CCSD iter', *cput1)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            break
    log.timer('CCSD', *cput0)
    return conv, eccsd, t1, t2


def update_amps(mycc, t1, t2, eris):
    if mycc.cc2:
        raise NotImplementedError
    assert(isinstance(eris, _ChemistsERIs))

    time0 = time.clock(), time.time()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    nov = nocc*nvir
    fock = eris.fock

    t1T = t1.T
    t2T = numpy.asarray(t2.transpose(2,3,0,1), order='C')
    t1 = t2 = None
    ntasks = mpi.pool.size
    vlocs = [_task_location(nvir, task_id) for task_id in range(ntasks)]
    vloc0, vloc1 = vlocs[rank]

    def _rotate_vir_block(buf):
        for task_id, buf in _rotate_tensor_block(buf):
            loc0, loc1 = vlocs[task_id]
            yield task_id, buf, loc0, loc1

    fswap = lib.H5TmpFile()
    wVooV = numpy.zeros((vloc1-vloc0,nocc,nocc,nvir))
    eris_voov = numpy.asarray(eris.ovvo).transpose(1,0,3,2)
    tau  = t2T * .5
    tau += numpy.einsum('ai,bj->abij', t1T[vloc0:vloc1], t1T)
    for task_id, tau, p0, p1 in _rotate_vir_block(tau):
        wVooV += lib.einsum('bkic,cajk->bija', eris_voov[:,:,:,p0:p1], tau)
    fswap['wVooV1'] = wVooV
    wVooV = tau = None

    eris_VOov = eris_voov - eris_voov.transpose(0,2,1,3)*.5
    eris_voov = None
    tau  = t2T.transpose(2,0,3,1) * 2 - t2T.transpose(3,0,2,1)
    tau -= numpy.einsum('ai,bj->jaib', t1T[vloc0:vloc1]*2, t1T)
    wVOov = eris_VOov.copy()
    for task_id, tau, p0, p1 in _rotate_vir_block(tau):
        wVOov += lib.einsum('dlkc,kcjb->dljb', eris_VOov[:,:,:,p0:p1], tau)
    fswap['wVOov1'] = wVOov
    wVOov = tau = eris_VOov = None

    t1Tnew = numpy.zeros_like(t1T)
    t2Tnew = _add_vvvv(mycc, t1T, t2T, eris, t2sym='jiba')
    t2Tnew *= .5  # *.5 because t2+t2.transpose(1,0,3,2) in the end
    time1 = log.timer_debug1('vvvv', *time0)

#** make_inter_F
    fov = fock[:nocc,nocc:].copy()
    t1Tnew += fock[nocc:,:nocc]

    foo = fock[:nocc,:nocc].copy()
    foo[numpy.diag_indices(nocc)] = 0
    foo += .5 * numpy.einsum('ai,ja->ji', fock[:nocc,nocc:], t1T)

    fvv = fock[nocc:,nocc:].copy()
    fvv[numpy.diag_indices(nvir)] = 0
    fvv -= .5 * numpy.einsum('ai,ib->ab', t1T, fock[:nocc,nocc:])

    foo_priv = numpy.zeros_like(foo)
    fvv_priv = numpy.zeros_like(fvv)
    t1T_priv = numpy.zeros_like(t1T)

    max_memory = mycc.max_memory - lib.current_memory()[0]
    unit = nocc*nvir**2*3 + nocc**2*nvir + 1
    blksize = min(nvir, max(BLKMIN, int((max_memory*.9e6/8-t2.size)/unit)))
    log.debug1('pass 1, max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d',
               max_memory, nocc, nvir, blksize)
    nvir_pair = nvir * (nvir+1) // 2
    def load_ovvv(p0, p1, buf):
        if p0 < p1:
            buf[:p1-p0] = eris.ovvv[:,p0:p1].transpose(1,0,2)

    fwVOov = fswap.create_dataset('wVOov', (vloc1-vloc0,nocc,nocc,nvir), 'f8')
    fwVooV = fswap.create_dataset('wVooV', (vloc1-vloc0,nocc,nocc,nvir), 'f8')

    buf = numpy.empty((blksize,nocc,nvir_pair))
    with lib.call_in_background(load_ovvv) as prefetch:
        load_ovvv(0, blksize, buf)
        for p0, p1 in lib.prange(vloc0, vloc1, blksize):
            i0, i1 = p0 - vloc0, p1 - vloc0
            eris_vovv, buf = buf[:p1-p0], numpy.empty_like(buf)
            prefetch(p1, min(vloc1, p1+blksize), buf)

            eris_vovv = lib.unpack_tril(eris_vovv.reshape((p1-p0)*nocc,nvir_pair))
            eris_vovv = eris_vovv.reshape(p1-p0,nocc,nvir,nvir)

            fvv_priv += 2*numpy.einsum('ck,ckab->ab', t1T[p0:p1], eris_vovv)
            fvv[:,p0:p1] -= numpy.einsum('ck,bkca->ab', t1T, eris_vovv)

            # Partition on index 0?
            vovv = alltoall(eris_vovv.transpose(3,1,0,2))
            if not mycc.direct:
                tau = t2T[p0:p1] + numpy.einsum('ai,bj->abij', t1T[p0:p1], t1T)
                for task_id, tau in _rotate_tensor_block(tau):
                    tmp = lib.einsum('bkcd,cdij->bkij', vovv[task_id], tau)
                    t2new -= lib.einsum('ka,bkij->baji', t1, tmp)
                tau = tmp = None

            #wVooV = numpy.zeros((p1-p0,nocc,nocc,nvir))
            wVooV = 0
            locs = comm.allgather((p0,p1))
            for task_id in range(ntasks):
                q0, q1 = locs[task_id]
                wVooV -= numpy.einsum('jc,ciba->bjia', t1[:,q0:q1], vovv)

            theta = t2T[i0:i1].transpose(0,2,1,3) * 2
            theta -= t2T[i0:i1].transpose(0,3,1,2)
            t1T_priv += lib.einsum('cjbi,cjba->ai', theta, eris_vovv)
            theta = None
            time1 = log.timer_debug1('vovv [%d:%d]'%(p0, p1), *time1)

            wVOov = lib.einsum('biac,jc->bija', eris_vovv, t1)
            fwVooV[i0:i1] = wVooV
            fwVOov[i0:i1] = wVOov
            eris_voov = eris_VOov = None

    time1 = log.timer_debug1('ovvv', *time1)

    unit = nocc**2*nvir*7 + nocc**3 + nocc*nvir**2
    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    blksize = min(nvir, max(BLKMIN, int((max_memory*.9e6/8-nocc**4)/unit)))
    log.debug1('pass 2, max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d',
               max_memory, nocc, nvir, blksize)

    woooo = numpy.zeros((nocc,nocc,nocc,nocc))

    for p0, p1 in lib.prange(vloc0, vloc1, blksize):
        i0, i1 = p0 - vloc0, p1 - vloc0
        wVOov = fwVOov[i0:i1]
        wVooV = fwVooV[i0:i1]
        eris_ovoo = eris.ovoo[:,p0:p1]
        foo += numpy.einsum('ck,kcji->ij', 2*t1T[p0:p1], eris_ovoo)
        foo += numpy.einsum('ck,icjk->ij',  -t1T[p0:p1], eris_ovoo)
        tmp = lib.einsum('al,jaik->lkji', t1T[p0:p1], eris_ovoo)
        woooo += tmp + tmp.transpose(1,0,3,2)
        tmp = None

        wVOov -= lib.einsum('jbik,ak->bjia', eris_ovoo, t1T)
        t2Tnew[p0:p1] += wVOov.transpose(0,3,1,2)

        wVooV += lib.einsum('kbij,ak->bija', eris_ovoo, t1T)
        eris_ovoo = None

        eris_oovv = eris.oovv[:,:,p0:p1]
        t1Tnew[p0:p1] -= numpy.einsum('bj,jiab->ai', t1T, eris_oovv)
        wVooV -= eris_oovv.transpose(2,0,1,3)

        eris_voov = eris.ovvo[:,p0:p1].transpose(1,0,3,2)
        t2Tnew[i0:i1] += eris_voov.transpose(0,3,1,2) * .5
        t1Tnew[p0:p1] += 2*numpy.einsum('bj,aijb->ai', t1T, eris_voov)

        tmp  = lib.einsum('ci,kjbc->bijk', t1T, eris_oovv)
        tmp += lib.einsum('bjkc,ci->bjik', eris_voov, t1T)
        t2new[p0:p1] -= numpy.einsum('bjik,ak->baji', tmp, t1T)
        eris_oovv = tmp = None

        fov[p0:p1] += numpy.einsum('kc,aikc->ia', t1T, eris_voov) * 2
        fov[p0:p1] -= numpy.einsum('kc,akic->ia', t1T, eris_voov)

        tau  = numpy.einsum('ai,bj->abij', t1T[p0:p1]*.5, t1T)
        tau += t2T[p0:p1]
        theta  = tau.transpose(0,1,3,2) * 2
        theta -= tau
        fvv_priv -= lib.einsum('caij,cjib->ab', theta.transpose(0,3,2,1), eris_voov)
        foo_priv += lib.einsum('aikb,kjab->ij', eris_voov, theta)
        tau = theta = None

        wVOov += wVooV*.5  #: bjia + bija*.5

        tau = t2[p0:p1] + numpy.einsum('ai,bj->abij', t1T[p0:p1], t1T)
        woooo += lib.einsum('abij,aklb->ijkl', tau, eris_voov)
        tau = None

        wVooV += fswap['wVooV1'][i0:i1]
        fwVooV[i0:i1] = wVooV
        wVOov += fswap['wVOov1'][i0:i1] * .5
        fwVOov[i0:i1] = wVOov
        eris_VOov = wVOov = wVooV = None
        time1 = log.timer_debug1('voov [%d:%d]'%(p0, p1), *time1)

    wVooV = numpy.asarray(fwVooV)
    for task_id, wVooV, p0, p1 in _rotate_vir_block(wVooV):
        tmp = lib.einsum('ackj,ckib->ajbi', t2[:,p0:p1], wVooV)
        t2Tnew += tmp.transpose(0,2,3,1)
        t2Tnew += tmp.transpose(0,2,1,3) * .5
    wVooV = tmp = None

    wVOov = numpy.asarray(fwVOov)
    theta  = t2 * 2
    theta -= t2.transpose(0,1,3,2)
    for task_id, wVOov, p0, p1 in _rotate_vir_block(wVOov):
        t2newT += lib.einsum('acik,ckjb->abij', theta[:,p0:p1], wVOov)
    wVOov = theta = None
    fwVOov = fwVooV = fswap = None

    theta = t2T.transpose(0,1,3,2) * 2 - t2T
    t1T_priv[vloc0:vloc1] += numpy.einsum('jb,abji->ai', fov, theta)
    ovoo = eris.ovoo[:,vloc0:vloc1]
    for task_id, ovoo, p0, p1 in _rotate_vir_block(ovoo):
        t1T_priv[p0:p1] -= lib.einsum('jbki,abjk->ai', ovoo, theta)
    theta = ovoo = None

    woooo = mpi.allreduce(woooo)
    woooo += numpy.asarray(eris.oooo).transpose(0,2,1,3).copy()
    tau = t2T + numpy.einsum('ai,bj->abij', t1T[vloc0:vloc1], t1T)
    t2Tnew += .5 * lib.einsum('abkl,klij->abij', tau, woooo)
    tau = woooo = None

    fvv += mpi.allreduce(fvv_priv)
    t1Tnew += mpi.allreduce(t1T_priv)

    ft_ij = foo + numpy.einsum('aj,ia->ij', .5*t1T, fov)
    ft_ab = fvv - numpy.einsum('ai,ib->ab', .5*t1T, fov)
    t2Tnew += lib.einsum('acij,bc->ijab', t2T, ft_ab)
    t2Tnew -= lib.einsum('ki,abkj->ijab', ft_ij, t2T)

    mo_e = fock.diagonal()
    eia = mo_e[:nocc,None] - mo_e[None,nocc:]
    t1Tnew += numpy.einsum('bi,ab->ai', t1T, fvv)
    t1Tnew -= numpy.einsum('aj,ji->ai', t1T, foo)
    t1Tnew /= eia.T

    t2tmp = mpi.alltoall([t2Tnew[p0:p1] for p0,p1 in vlocs])
    for task_id, tmp in enumerate(t2tmp):
        loc0, loc1 = vlocs[task_id]
        t2Tnew[:,loc0:loc1] = tmp.transpose(1,0,3,2)

    for i in range(loc0, loc1):
        t2Tnew[i-loc0] /= lib.direct_sum('a+jb->abj', eia[:,i], eia)

    time0 = log.timer_debug1('update t1 t2', *time0)
    return t1new, t2new

def _add_vvvv(mycc, t1T, t2T, eris, out=None, with_ovvv=None, t2sym=None):
    '''t2sym: whether t2 has the symmetry t2[ijab]==t2[jiba] or
    t2[ijab]==-t2[jiab] or t2[ijab]==-t2[jiba]
    '''
    if t2sym == 'jiba':
        nvir0, nvir, nocc = t2T.shape[:3]
        Ht2tril = _add_vvvv_tril(mycc, t1T, t2T, eris, with_ovvv=with_ovvv)
        nocc, nvir = t2.shape[1:3]
        #?Ht2 = _unpack_t2_tril(Ht2tril, nocc, nvir, out, t2sym)
        Ht2 = unpack_tril(Ht2tril.reshape(nvir0,nvir,nocc*(nocc+1)//2))
    else:
        Ht2 = _add_vvvv_full(mycc, t1T, t2T, eris, out, with_ovvv)
    return Ht2

def _add_vvvv_tril(mycc, t1, t2, eris, out=None, with_ovvv=None):
    '''Ht2 = numpy.einsum('ijcd,acdb->ijab', t2, vvvv)
    Using symmetry t2[ijab] = t2[jiba] and Ht2[ijab] = Ht2[jiba], compute the
    lower triangular part of  Ht2
    '''
    time0 = time.clock(), time.time()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    if with_ovvv is None:
        with_ovvv = mycc.direct
    nocc, nvir_seg, nvir = t2.shape[1:4]
    mo_loc0, mo_loc1 = _task_location(nvir, rank)
    nocc2 = nocc*(nocc+1)//2
    if t1 is None:
        tau = t2[numpy.tril_indices(nocc)]
    else:
        tau = numpy.empty((nocc2,mo_loc1-mo_loc0,nvir), dtype=t2.dtype)
        p1 = 0
        for i in range(nocc):
            p0, p1 = p1, p1 + i+1
            tau[p0:p1] = numpy.einsum('a,jb->jab', t1[i,mo_loc0:mo_loc1], t1[:i+1])
            tau[p0:p1] += t2[i,:i+1]

    if mycc.direct:   # AO-direct CCSD
        mo = getattr(eris, 'mo_coeff', None)
        if mo is None:  # If eris does not have the attribute mo_coeff
            mo = _mo_without_core(mycc, mycc.mo_coeff)

        tau_shape = tau.shape
        ao_loc = mol.ao_loc_nr()
        orbv = mo[:,nocc:]
        nao, nvir = orbv.shape

        ntasks = mpi.pool.size
        task_sh_locs = lib.misc._balanced_partition(ao_loc, ntasks)
        ao_loc0 = ao_loc[task_sh_locs[rank  ]]
        ao_loc1 = ao_loc[task_sh_locs[rank+1]]

        tau = lib.einsum('xab,pb->xap', tau, orbv)
        tau_priv = numpy.zeros((nocc2,ao_loc1-ao_loc0,nao))
        for task_id, tau in _rotate_tensor_block(tau):
            mo_loc0, mo_loc1 = _task_location(nvir, task_id)
            tau_priv += lib.einsum('xab,pa->xpb',
                                   tau, orbv[ao_loc0:ao_loc1,mo_loc0:mo_loc1])
        tau = None
        time1 = log.timer_debug1('vvvv-tau mo2ao', *time0)

        buf = _contract_vvvv_t2(mycc, None, tau_priv, task_sh_locs, None,
                                max_memory, log)
        buf = buf_ao = buf.reshape(tau_priv.shape)
        tau_priv = None
        time1 = log.timer_debug1('vvvv-tau contraction', *time1)

        mo_loc0, mo_loc1 = _task_location(nvir, rank)
        buf = lib.einsum('xap,pb->xab', buf, orbv)
        Ht2tril = numpy.ndarray((nocc2,nvir_seg,nvir), buffer=out)
        Ht2tril[:] = 0
        for task_id, buf in _rotate_tensor_block(buf):
            ao_loc0 = ao_loc[task_sh_locs[task_id  ]]
            ao_loc1 = ao_loc[task_sh_locs[task_id+1]]
            Ht2tril += lib.einsum('xpb,pa->xab',
                                  buf, orbv[ao_loc0:ao_loc1,mo_loc0:mo_loc1])

        time1 = log.timer_debug1('vvvv-tau ao2mo', *time1)

        if with_ovvv:
            #: tmp = numpy.einsum('ijcd,ka,kdcb->ijba', tau, t1, eris.ovvv)
            #: t2new -= tmp + tmp.transpose(1,0,3,2)
            orbo = mo[:,:nocc]
            buf = lib.einsum('xap,pb->xab', buf_ao, orbo)
            tmp = numpy.zeros((nocc2,nvir_seg,nocc))
            for task_id, buf in _rotate_tensor_block(buf):
                ao_loc0 = ao_loc[task_sh_locs[task_id  ]]
                ao_loc1 = ao_loc[task_sh_locs[task_id+1]]
                tmp += lib.einsum('xpb,pa->xab',
                                  buf, orbv[ao_loc0:ao_loc1,mo_loc0:mo_loc1])
            Ht2tril -= lib.ddot(tmp.reshape(nocc2*nvir_seg,nocc), t1).reshape(nocc2,nvir,nvir)
            tmp = None

            mo_loc0, mo_loc1 = _task_location(nvir, rank)
            t1_ao = numpy.dot(orbo, t1)
            buf = lib.einsum('xap,pb->xab', buf_ao, orbv)
            for task_id, buf in _rotate_tensor_block(buf):
                ao_loc0 = ao_loc[task_sh_locs[task_id  ]]
                ao_loc1 = ao_loc[task_sh_locs[task_id+1]]
                Ht2tril -= lib.einsum('xpb,pa->xab',
                                      buf, t1_ao[ao_loc0:ao_loc1,mo_loc0:mo_loc1])

        time1 = log.timer_debug1('vvvv-tau ao2mo', *time0)
    else:
        raise NotImplementedError
        assert(not with_ovvv)
        max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
        #?Ht2tril = _contract_vvvv_t2(mycc, ftau, mycc.direct, out, max_memory, log)
        #buf = _contract_vvvv_t2(mycc, eris.vvvv, tau, task_locs, out, max_memory, log)
    return Ht2tril

def _task_location(n, task=rank):
    ntasks = mpi.pool.size
    seg_size = (n + ntasks - 1) // ntasks
    loc0 = seg_size * task
    loc1 = min(n, loc0 + seg_size)
    return loc0, loc1

def _rotate_tensor_block(fbuf):
    buf = fbuf()
    ntasks = mpi.pool.size
    tasks = list(range(ntasks))
    tasks = tasks[rank:] + tasks[:rank]
    for task in tasks:
        if task != rank:
            buf = mpi.rotate(buf)
        yield task, buf

def _add_vvvv_full(mycc, t1T, t2T, eris, out=None, with_ovvv=False):
    '''Ht2 = numpy.einsum('ijcd,acdb->ijab', t2, vvvv)
    without using symmetry t2[ijab] = t2[jiba] in t2 or Ht2
    '''
    time0 = time.clock(), time.time()
    log = logger.Logger(mycc.stdout, mycc.verbose)

    nocc, nvir_seg, nvir = t2.shape[1:4]
    mo_loc0, mo_loc1 = _task_location(nvir, rank)
    if t1 is None:
        tau = t2
    else:
        tau = numpy.einsum('ia,jb->ijab', t1[:,mo_loc0:mo_loc1], t1)
        tau += t2
    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])

    if mycc.direct:   # AO-direct CCSD
        if with_ovvv:
            raise NotImplementedError
        mo = getattr(eris, 'mo_coeff', None)
        if mo is None:  # If eris does not have the attribute mo_coeff
            mo = _mo_without_core(mycc, mycc.mo_coeff)

#X        orbv = mo[:,nocc:]
#X        task_sh_locs = lib.misc._balanced_partition(ao_loc, ntasks)
#X        Ht2 = _ao_direct_contract_vvvv(mol, ftau, orbv, task_sh_locs, out,
#X                                       max_memory, log)
#X
        ao_loc = mol.ao_loc_nr()
        nao, nmo = mo.shape

        ntasks = mpi.pool.size
        task_sh_locs = lib.misc._balanced_partition(ao_loc, ntasks)
        ao_loc0 = ao_loc[task_sh_locs[rank  ]]
        ao_loc1 = ao_loc[task_sh_locs[rank+1]]

        orbv = mo[:,nocc:]
        tau = lib.einsum('ijab,pb->ijap', tau, orbv)
        tau_priv = numpy.zeros((nocc,nocc,ao_loc1-ao_loc0,nao))
        for task_id, tau in _rotate_tensor_block(tau):
            mo_loc0, mo_loc1 = _task_location(nvir, task_id)
            tau_priv += lib.einsum('ijab,pa->ijpb',
                                   tau, orbv[ao_loc0:ao_loc1,mo_loc0:mo_loc1])
        tau = None
        time1 = log.timer_debug1('vvvv-tau mo2ao', *time0)

        buf = _contract_vvvv_t2(mycc, None, tau_priv, task_sh_locs, None,
                                max_memory, log)
        buf = buf.reshape(tau_priv.shape)
        tau_priv = None
        time1 = log.timer_debug1('vvvv-tau contraction', *time1)

        mo_loc0, mo_loc1 = _task_location(nvir, rank)
        buf = lib.einsum('ijap,pb->ijab', buf, orbv)
        Ht2 = numpy.ndarray(t2.shape, buffer=out)
        Ht2[:] = 0
        for task_id, buf in _rotate_tensor_block(buf):
            ao_loc0 = ao_loc[task_sh_locs[task_id  ]]
            ao_loc1 = ao_loc[task_sh_locs[task_id+1]]
            Ht2 += lib.einsum('ijpb,pa->ijab',
                              buf, orbv[ao_loc0:ao_loc1,mo_loc0:mo_loc1])

        time1 = log.timer_debug1('vvvv-tau ao2mo', *time1)

    else:
        raise NotImplementedError
        #assert(not with_ovvv)
        #Ht2 = _contract_vvvv_t2(mycc, eris.vvvv, tau, task_locs, out, max_memory, log)

    return Ht2.reshape(t2.shape)

#Xdef _ao_direct_contract_vvvv(mol, ftau, orbv, task_locs, out, max_memory, log):
#X    time0 = time.clock(), time.time()
#X    tau = ftau()
#X    tau_shape = tau.shape
#X    ao_loc = mol.ao_loc_nr()
#X    nao, nvir = orbv.shape
#X
#X    ntasks = mpi.pool.size
#X    task_sh_locs = task_locs
#X    ao_loc0 = ao_loc[task_sh_locs[rank  ]]
#X    ao_loc1 = ao_loc[task_sh_locs[rank+1]]
#X
#X    tau = lib.einsum('xab,pb->xap', tau, orbv)
#X    tau_priv = numpy.zeros((tau_shape[0],ao_loc1-ao_loc0,nao))
#X    for task in range(ntasks):
#X        if task != rank:
#X            tau = mpi.rotate(tau)
#X        mo_loc0, mo_loc1 = _task_location(nvir, task)
#X
#X        tau_priv += lib.einsum('xab,pa->xpb',
#X                               tau, orbv[ao_loc0:ao_loc1,mo_loc0:mo_loc1])
#X    tau = None
#X    time1 = log.timer_debug1('vvvv-tau mo2ao', *time0)
#X
#X    buf = _contract_vvvv_t2(mycc, None, tau_priv, task_sh_locs, None,
#X                            max_memory, log)
#X    buf = buf.reshape(tau_priv.shape)
#X    tau_priv = None
#X    time1 = log.timer_debug1('vvvv-tau contraction', *time1)
#X
#X    mo_loc0, mo_loc1 = _task_location(nvir, rank)
#X    buf = lib.einsum('xap,pb->xab', buf, orbv)
#X    Ht2 = numpy.ndarray(tau_shape, buffer=out)
#X    Ht2[:] = 0
#X    for task in range(ntasks):
#X        if task != rank:
#X            buf = mpi.rotate(buf)
#X        ao_loc0 = ao_loc[task_sh_locs[task  ]]
#X        ao_loc1 = ao_loc[task_sh_locs[task+1]]
#X
#X        Ht2 += lib.einsum('xpb,pa->xab',
#X                          buf, orbv[ao_loc0:ao_loc1,mo_loc0:mo_loc1])
#X
#X    time1 = log.timer_debug1('vvvv-tau ao2mo', *time1)
#X    return Ht2


def _contract_vvvv_t2(mycc, vvvv, t2, task_locs, out=None, max_memory=MEMORYMIN,
                      verbose=None):
    '''Ht2 = numpy.einsum('ijcd,acbd->ijab', t2, vvvv)
    where vvvv has to be real and has the 4-fold permutation symmetry

    Args:
        vvvv : None or integral object
            if vvvv is None, contract t2 to AO-integrals using AO-direct algorithm
    '''
    time0 = time.clock(), time.time()
    mol = mycc.mol
    log = logger.new_logger(mycc, verbose)

    if callable(t2):
        x2 = t2()
    else:
        x2 = t2
    assert(x2.dtype == numpy.double)
    nvira, nvirb = x2.shape[-2:]
    nvir2 = nvira * nvirb
    x2 = x2.reshape(-1,nvira,nvirb)
    nocc2 = x2.shape[0]
    Ht2 = numpy.ndarray(x2.shape, dtype=x2.dtype, buffer=out)
    Ht2[:] = 0

    _dgemm = lib.numpy_helper._dgemm
    def contract_blk_(Ht2, x2, eri, i0, i1, j0, j1):
        ic = i1 - i0
        jc = j1 - j0
        #:Ht2[:,j0:j1] += numpy.einsum('xef,efab->xab', x2[:,i0:i1], eri)
        _dgemm('N', 'N', nocc2, jc*nvirb, ic*nvirb,
               x2.reshape(-1,nvir2), eri.reshape(-1,jc*nvirb),
               Ht2.reshape(-1,nvir2), 1, 1, i0*nvirb, 0, j0*nvirb)

    if vvvv is None:   # AO-direct CCSD
        ao_loc = mol.ao_loc_nr()
        assert(nvira == nvirb == ao_loc[-1])

        intor = mol._add_suffix('int2e')
        ao2mopt = _ao2mo.AO2MOpt(mol, intor, 'CVHFnr_schwarz_cond',
                                 'CVHFsetnr_direct_scf')
        max_words = max(0, max_memory*.95e6/8-t2.size)
        blksize = max(BLKMIN, numpy.sqrt(max_words/nvirb**2/2))
        fint = gto.moleintor.getints4c

        ntasks = mpi.pool.size
        task_sh_locs = task_locs
        sh_ranges_tasks = []
        for task in range(ntasks):
            sh0 = task_sh_locs[task]
            sh1 = task_sh_locs[task+1]
            sh_ranges = ao2mo.outcore.balance_partition(ao_loc, blksize, sh0, sh1-1)
            sh_ranges_tasks.append(sh_ranges)

        blksize = max(max(x[2] for x in sh_ranges)
                      for sh_ranges in sh_ranges_tasks)
        eribuf = numpy.empty((blksize,blksize,nvirb,nvirb))
        loadbuf = numpy.empty((blksize,blksize,nvirb,nvirb))

        out_sh_ranges = sh_ranges_tasks[rank]
        out_sh0 = task_sh_locs[rank]
        out_sh1 = task_sh_locs[rank+1]
        out_offset = ao_loc[out_sh0]
        assert(nvira == ao_loc[out_sh1] - ao_loc[out_sh0])

        for task_id, x2 in _rotate_tensor_block(x2):
            sh_ranges = sh_ranges_tasks[task_id]
            sh0 = task_sh_locs[task_id]
            cur_offset = ao_loc[sh0]

            for ish0, ish1, ni in sh_ranges:
                for jsh0, jsh1, nj in out_sh_ranges:
                    eri = fint(intor, mol._atm, mol._bas, mol._env,
                               shls_slice=(ish0,ish1,jsh0,jsh1), aosym='s2kl',
                               ao_loc=ao_loc, cintopt=ao2mopt._cintopt, out=eribuf)
                    i0, i1 = ao_loc[ish0] - cur_offset, ao_loc[ish1] - cur_offset
                    j0, j1 = ao_loc[jsh0] - out_offset, ao_loc[jsh1] - out_offset
                    tmp = numpy.ndarray((i1-i0,nvirb,j1-j0,nvirb), buffer=loadbuf)
                    _ccsd.libcc.CCload_eri(tmp.ctypes.data_as(ctypes.c_void_p),
                                           eri.ctypes.data_as(ctypes.c_void_p),
                                           (ctypes.c_int*4)(i0, i1, j0, j1),
                                           ctypes.c_int(nvirb))
                    contract_blk_(x2, Ht2, tmp, i0, i1, j0, j1)
                    time0 = log.timer_debug1('AO-vvvv [%d:%d,%d:%d]' %
                                             (ish0,ish1,jsh0,jsh1), *time0)
    else:
        raise NotImplementedError
    return Ht2.reshape(t2.shape)

def _contract_s1vvvv_t2(mycc, mol, vvvv, t2, out=None, max_memory=MEMORYMIN,
                        verbose=None):
    '''Ht2 = numpy.einsum('ijcd,acdb->ijab', t2, vvvv)
    where vvvv can be real or complex and no permutation symmetry is available in vvvv.

    Args:
        vvvv : None or integral object
            if vvvv is None, contract t2 to AO-integrals using AO-direct algorithm
    '''
    # vvvv == None means AO-direct CCSD. It should redirect to
    # _contract_s4vvvv_t2(mycc, mol, vvvv, t2, out, max_memory, verbose)
    assert(vvvv is not None)

    time0 = time.clock(), time.time()
    log = logger.new_logger(mol, verbose)

    nvira, nvirb = t2.shape[-2:]
    x2 = t2.reshape(-1,nvira,nvirb)
    nocc2 = x2.shape[0]
    dtype = numpy.result_type(t2, vvvv)
    Ht2 = numpy.ndarray(x2.shape, dtype=dtype, buffer=out)

    unit = nvirb**2*nvira*2 + nocc2*nvirb + 1
    blksize = min(nvira, max(BLKMIN, int(max_memory*1e6/8/unit)))

    for p0,p1 in lib.prange(0, nvira, blksize):
        Ht2[:,p0:p1] = lib.einsum('xcd,acbd->xab', x2, vvvv[p0:p1])
        time0 = log.timer_debug1('vvvv [%d:%d]' % (p0,p1), *time0)
    return Ht2.reshape(t2.shape)

def _unpack_t2_tril(t2tril, nocc, nvir, out=None, t2sym='jiba'):
    t2 = numpy.ndarray((nocc,nocc,nvir,nvir), dtype=t2tril.dtype, buffer=out)
    idx,idy = numpy.tril_indices(nocc)
    if t2sym == 'jiba':
        t2[idy,idx] = t2tril.transpose(0,2,1)
        t2[idx,idy] = t2tril
    elif t2sym == '-jiba':
        t2[idy,idx] = -t2tril.transpose(0,2,1)
        t2[idx,idy] = t2tril
    elif t2sym == '-jiab':
        t2[idy,idx] =-t2tril
        t2[idx,idy] = t2tril
        t2[numpy.diag_indices(nocc)] = 0
    return t2

def _unpack_4fold(c2vec, nocc, nvir, anti_symm=True):
    t2 = numpy.zeros((nocc**2,nvir**2), dtype=c2vec.dtype)
    if nocc > 1 and nvir > 1:
        t2tril = c2vec.reshape(nocc*(nocc-1)//2,nvir*(nvir-1)//2)
        otril = numpy.tril_indices(nocc, k=-1)
        vtril = numpy.tril_indices(nvir, k=-1)
        lib.takebak_2d(t2, t2tril, otril[0]*nocc+otril[1], vtril[0]*nvir+vtril[1])
        lib.takebak_2d(t2, t2tril, otril[1]*nocc+otril[0], vtril[1]*nvir+vtril[0])
        if anti_symm:  # anti-symmetry when exchanging two particle indices
            t2tril = -t2tril
        lib.takebak_2d(t2, t2tril, otril[0]*nocc+otril[1], vtril[1]*nvir+vtril[0])
        lib.takebak_2d(t2, t2tril, otril[1]*nocc+otril[0], vtril[0]*nvir+vtril[1])
    return t2.reshape(nocc,nocc,nvir,nvir)

def amplitudes_to_vector(t1, t2, out=None):
    nocc, nvir = t1.shape
    nov = nocc * nvir
    size = nov + nov*(nov+1)//2
    vector = numpy.ndarray(size, t1.dtype, buffer=out)
    vector[:nov] = t1.ravel()
    lib.pack_tril(t2.transpose(0,2,1,3).reshape(nov,nov), out=vector[nov:])
    return vector

def vector_to_amplitudes(vector, nmo, nocc):
    nvir = nmo - nocc
    nov = nocc * nvir
    t1 = vector[:nov].copy().reshape((nocc,nvir))
    # filltriu=lib.SYMMETRIC because t2[iajb] == t2[jbia]
    t2 = lib.unpack_tril(vector[nov:], filltriu=lib.SYMMETRIC)
    t2 = t2.reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3)
    return t1, numpy.asarray(t2, order='C')

def amplitudes_to_vector_s4(t1, t2, out=None):
    nocc, nvir = t1.shape
    nov = nocc * nvir
    size = nov + nocc*(nocc-1)//2*nvir*(nvir-1)//2
    vector = numpy.ndarray(size, t1.dtype, buffer=out)
    vector[:nov] = t1.ravel()
    otril = numpy.tril_indices(nocc, k=-1)
    vtril = numpy.tril_indices(nvir, k=-1)
    lib.take_2d(t2.reshape(nocc**2,nvir**2), otril[0]*nocc+otril[1],
                vtril[0]*nvir+vtril[1], out=vector[nov:])
    return vector

def vector_to_amplitudes_s4(vector, nmo, nocc):
    nvir = nmo - nocc
    nov = nocc * nvir
    size = nov + nocc*(nocc-1)//2*nvir*(nvir-1)//2
    t1 = vector[:nov].copy().reshape(nocc,nvir)
    t2 = numpy.zeros((nocc,nocc,nvir,nvir), dtype=vector.dtype)
    t2 = _unpack_4fold(vector[nov:size], nocc, nvir)
    return t1, t2


def energy(mycc, t1=None, t2=None, eris=None):
    '''CCSD correlation energy'''
    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    if eris is None: eris = mycc.ao2mo()

    nocc, nvir = t1.shape
    fock = eris.fock
    e = numpy.einsum('ia,ia', fock[:nocc,nocc:], t1) * 2
    max_memory = mycc.max_memory - lib.current_memory()[0]
    blksize = int(min(nvir, max(BLKMIN, max_memory*.3e6/8/(nocc**2*nvir+1))))
    for p0, p1 in lib.prange(0, nvir, blksize):
        eris_ovvo = eris.ovvo[:,p0:p1]
        tau = t2[:,:,p0:p1] + numpy.einsum('ia,jb->ijab', t1[:,p0:p1], t1)
        e += 2 * numpy.einsum('ijab,iabj', tau, eris_ovvo)
        e -=     numpy.einsum('jiab,iabj', tau, eris_ovvo)
    if abs(e.imag) > 1e-4:
        logger.warn(cc, 'Non-zero imaginary part found in CCSD energy %s', e)
    return e


def _init_ccsd(ccsd_obj):
    print rank
    time.sleep(5)
    if rank == 0:
        mol = comm.bcast(ccsd_obj.mol.dumps())
    else:
        ccsd_obj = CCSD.__new__(CCSD)
        ccsd_obj.mol = gto.mole.loads(comm.bcast())

    _sync(ccsd_obj)
    key = id(ccsd_obj)
    mpi._registry[key] = ccsd_obj
    regs = mpi.comm.gather(key)
    return regs
class CCSD(ccsd.CCSD):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        ccsd.CCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.direct = True
        regs = mpi.pool.apply(_init_ccsd, self, (None,))
        self._reg_procs = regs

    def pack(self):
        return {'verbose'   : self.verbose,
                'max_memory': self.max_memory,
                'frozen'    : self.frozen,
                '_nocc'     : self._nocc,
                '_nmo'      : self._nmo}
    def unpack_(self, ccdic):
        self.__dict__.update(ccdic)
        return self

    def dump_flags(self):
        if rank == 0:
            ccsd.CCSD.dump_flags(self)
        return self
    def sanity_check(self):
        if rank == 0:
            ccsd.CCSD.sanity_check(self)
        return self

    def get_init_guess(self, eris=None):
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        return self.init_amps(eris)[1:]
    def init_amps(self, eris):
        time0 = time.clock(), time.time()
        mo_e = eris.fock.diagonal()
        nocc = self.nocc
        nvir = mo_e.size - nocc
        eia = mo_e[:nocc,None] - mo_e[None,nocc:]
        t1 = eris.fock[:nocc,nocc:] / eia
        t2 = numpy.empty((nocc,nocc,nvir,nvir))
        max_memory = self.max_memory - lib.current_memory()[0]
        blksize = int(min(nvir, max(BLKMIN, max_memory*.3e6/8/(nocc**2*nvir+1))))
        self.emp2 = 0
        for p0, p1 in lib.prange(0, nvir, blksize):
            eris_ovvo = eris.ovvo[:,p0:p1]
            t2[:,:,p0:p1] = (eris_ovvo.transpose(0,3,1,2)
                             / lib.direct_sum('ia,jb->ijab', eia[:,p0:p1], eia))
            self.emp2 += 2 * numpy.einsum('ijab,iabj', t2[:,:,p0:p1], eris_ovvo)
            self.emp2 -=     numpy.einsum('jiab,iabj', t2[:,:,p0:p1], eris_ovvo)

        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        logger.timer(self, 'init mp2', *time0)
        return self.emp2, t1, t2

    energy = energy
    _add_vvvv = _add_vvvv
    update_amps = update_amps

    def kernel(self, t1=None, t2=None, eris=None):
        return self.ccsd(t1, t2, eris)
    def ccsd(self, t1=None, t2=None, eris=None):
        assert(self.mo_coeff is not None)
        assert(self.mo_occ is not None)
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        self.converged, self.e_corr, self.t1, self.t2 = \
                kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                       tol=self.conv_tol, tolnormt=self.conv_tol_normt,
                       verbose=self.verbose)
        if rank == 0:
            self._finalize()
        return self.e_corr, self.t1, self.t2

    def ao2mo(self, mo_coeff=None):
        return _make_eris_outcore(self, mo_coeff)

    def run_diis(self, t1, t2, istep, normt, de, adiis):
        if (adiis and
            istep >= self.diis_start_cycle and
            abs(de) < self.diis_start_energy_diff):
            vec = self.amplitudes_to_vector(t1, t2)
            t1, t2 = self.vector_to_amplitudes(adiis.update(vec))
            logger.debug1(self, 'DIIS for step %d', istep)
        return t1, t2

    def amplitudes_to_vector(self, t1, t2, out=None):
        return amplitudes_to_vector(t1, t2, out)

    def vector_to_amplitudes(self, vec, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        return vector_to_amplitudes(vec, nmo, nocc)

CC = RCCSD = CCSD


#FIXME:
class _ChemistsERIs:
    '''(pq|rs)'''
    def __init__(self, mol=None):
        self.mol = mol
        self.mo_coeff = None
        self.nocc = None
        self.fock = None

        self.oooo = None
        self.ovoo = None
        self.oovv = None
        self.ovvo = None
        self.ovov = None
        self.ovvv = None
        self.vvvv = None

    def _common_init_(self, mycc, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mycc.mo_coeff
        self.mo_coeff = mo_coeff = _mo_without_core(mycc, mo_coeff)
# Note: Recomputed fock matrix since SCF may not be fully converged.
        dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
        fockao = mycc._scf.get_hcore() + mycc._scf.get_veff(mycc.mol, dm)
        self.fock = reduce(numpy.dot, (mo_coeff.conj().T, fockao, mo_coeff))
        self.nocc = mycc.nocc
        self.mol = mycc.mol

        mo_e = self.fock.diagonal()
        try:
            gap = abs(mo_e[:self.nocc,None] - mo_e[None,self.nocc:]).min()
            if gap < 1e-5:
                logger.warn(mycc, 'HOMO-LUMO gap %s too small for CCSD', gap)
        except ValueError:  # gap.size == 0
            pass
        return self

    def get_ovvv(self, *slices):
        '''To access a subblock of ovvv tensor'''
        ovw = numpy.asarray(self.ovvv[slices])
        nocc, nvir, nvir_pair = ovw.shape
        ovvv = lib.unpack_tril(ovw.reshape(nocc*nvir,nvir_pair))
        nvir1 = ovvv.shape[2]
        return ovvv.reshape(nocc,nvir,nvir1,nvir1)

    def _contract_vvvv_t2(self, mycc, t2, vvvv_or_direct=False, out=None,
                          max_memory=MEMORYMIN, verbose=None):
        if isinstance(vvvv_or_direct, numpy.ndarray):
            vvvv = vvvv_or_direct
        elif vvvv_or_direct:  # AO-direct contraction
            vvvv = None
        else:
            vvvv = self.vvvv
        return _contract_vvvv_t2(mycc, self.mol, vvvv, t2, out, max_memory, verbose)

    def _contract_vvvv_oov(self, mycc, r2, out=None):
        raise NotImplementedError

    def _contract_vvvv_ovv(self, mycc, r2, out=None):
        raise NotImplementedError

@mpi.parallel_call
def _make_eris_outcore(mycc, mo_coeff=None):
    cput0 = (time.clock(), time.time())
    log = logger.Logger(mycc.stdout, mycc.verbose)
    eris = ccsd._ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    mol = mycc.mol
    mo_coeff = numpy.asarray(eris.mo_coeff, order='F')
    nocc = eris.nocc
    nao, nmo = mo_coeff.shape
    nvir = nmo - nocc
    orbo = mo_coeff[:,:nocc]
    orbv = mo_coeff[:,nocc:]
    nvpair = nvir * (nvir+1) // 2
    v0, v1 = _task_location(nvir)

    eris.feri1 = lib.H5TmpFile()
    eris.oooo = eris.feri1.create_dataset('oooo', (nocc,nocc,nocc,nocc), 'f8')
    eris.oovv = eris.feri1.create_dataset('oovv', (nocc,nocc,v1-v0,nvir), 'f8', chunks=(nocc,nocc,1,nvir))
    eris.ovoo = eris.feri1.create_dataset('ovoo', (nocc,v1-v0,nocc,nocc), 'f8', chunks=(nocc,1,nocc,nocc))
    eris.ovvo = eris.feri1.create_dataset('ovvo', (nocc,v1-v0,nvir,nocc), 'f8', chunks=(nocc,1,nvir,nocc))
    eris.ovov = eris.feri1.create_dataset('ovov', (nocc,v1-v0,nocc,nvir), 'f8', chunks=(nocc,1,nocc,nvir))
    eris.ovvv = eris.feri1.create_dataset('ovvv', (nocc,v1-v0,nvpair), 'f8', chunks=(nocc,1,nvpair))
    assert(mycc.direct)

    oovv = numpy.empty((nocc,nocc,nvir,nvir))
    def save_occ_frac(p0, p1, eri):
        eri = eri.reshape(p1-p0,nocc,nmo,nmo)
        eris.oooo[p0:p1] = eri[:,:,:nocc,:nocc]
        eris.oovv[p0:p1] = eri[:,:,nocc+v0:nocc+v1,nocc:]

    def save_vir_frac(p0, p1, eri):
        eri = eri.reshape(p1-p0,nocc,nmo,nmo)
        eris.ovoo[:,p0:p1] = eri[:,:,:nocc,:nocc].transpose(1,0,2,3)
        eris.ovvo[:,p0:p1] = eri[:,:,nocc:,:nocc].transpose(1,0,2,3)
        eris.ovov[:,p0:p1] = eri[:,:,:nocc,nocc:].transpose(1,0,2,3)
        vvv = lib.pack_tril(eri[:,:,nocc:,nocc:].reshape((p1-p0)*nocc,nvir,nvir))
        eris.ovvv[:,p0:p1] = vvv.reshape(p1-p0,nocc,nvpair).transpose(1,0,2)

    cput1 = time.clock(), time.time()

    fswap = lib.H5TmpFile()
    max_memory = max(MEMORYMIN, mycc.max_memory-lib.current_memory()[0])
    int2e = mol._add_suffix('int2e')
    orbov = numpy.hstack((orbo, orbv[:,v0:v1]))
    ao2mo.outcore.half_e1(mol, (orbov,orbo), fswap, int2e,
                          's4', 1, max_memory, verbose=log)

    ao_loc = mol.ao_loc_nr()
    nao_pair = nao * (nao+1) // 2
    blksize = int(min(8e9,max_memory*.5e6)/8/(nao_pair+nmo**2)/nocc)
    blksize = min(nmo, max(BLKMIN, blksize))
    fload = ao2mo.outcore._load_from_h5g
    def prefetch(p0, p1, rowmax, buf):
        p0, p1 = p1, min(rowmax, p1+blksize)
        if p0 < p1:
            fload(fswap['0'], p0*nocc, p1*nocc, buf)

    buf = numpy.empty((blksize*nocc,nao_pair))
    buf_prefetch = numpy.empty_like(buf)
    outbuf = numpy.empty((blksize*nocc,nmo**2))
    with lib.call_in_background(prefetch) as bprefetch:
        fload(fswap['0'], 0, min(nocc,blksize)*nocc, buf_prefetch)
        for p0, p1 in lib.prange(0, nocc, blksize):
            nrow = (p1 - p0) * nocc
            buf, buf_prefetch = buf_prefetch, buf
            bprefetch(p0, p1, nocc, buf_prefetch)
            dat = ao2mo._ao2mo.nr_e2(buf[:nrow], mo_coeff, (0,nmo,0,nmo),
                                     's4', 's1', out=outbuf, ao_loc=ao_loc)
            save_occ_frac(p0, p1, dat)

        norb_max = nocc + v1 - v0
        fload(fswap['0'], nocc**2, min(nocc+blksize,norb_max)*nocc, buf_prefetch)
        for p0, p1 in lib.prange(0, v1-v0, blksize):
            nrow = (p1 - p0) * nocc
            buf, buf_prefetch = buf_prefetch, buf
            bprefetch(nocc+p0, nocc+p1, norb_max, buf_prefetch)
            dat = ao2mo._ao2mo.nr_e2(buf[:nrow], mo_coeff, (0,nmo,0,nmo),
                                     's4', 's1', out=outbuf, ao_loc=ao_loc)
            save_vir_frac(p0, p1, dat)

    cput1 = log.timer_debug1('transforming oppp', *cput1)
    log.timer('CCSD integral transformation', *cput0)
    return eris

def _sync(mycc):
    return mycc.unpack_(comm.bcast(mycc.pack()))


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import cc

    mol = gto.Mole()
    mol.atom = [
        [2 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = '6-31g'
    mol.build()
    mf = scf.RHF(mol)
    nao = mol.nao_nr()
    numpy.random.seed(1)
    mf.mo_coeff = numpy.random.random((nao,nao)) - 0.5
    mf.mo_occ = numpy.zeros(nao)
    nocc = mol.nelectron // 2
    nvir = nao - nocc
    mf.mo_occ[:mol.nelectron//2] = 2

    mycc = cc.CCSD(mf)
    eris = mycc.ao2mo(mf.mo_coeff)
    mycc1 = CCSD(mf)
    eris1 = _make_eris_outcore(mycc1, mf.mo_coeff)
    print 'kkkkkkkkk '
    oovv = mpi.gather(numpy.asarray(eris1.oovv).transpose(2,0,1,3)).transpose(1,2,0,3)
    print 'kkkkkkkkk0'
    ovvo = mpi.gather(numpy.asarray(eris1.ovvo).transpose(1,0,2,3)).transpose(1,0,2,3)
    print 'kkkkkkkkk1'
    ovov = mpi.gather(numpy.asarray(eris1.ovov).transpose(1,0,2,3)).transpose(1,0,2,3)
    print 'kkkkkkkkk2'
    if rank == 0:
        print(abs(numpy.asarray(eris1.oooo) - numpy.asarray(eris.oooo)).max())
        print(abs(numpy.asarray(oovv) - numpy.asarray(eris.oovv)).max())
        print(abs(numpy.asarray(ovvo) - numpy.asarray(eris.ovvo)).max())
        print(abs(numpy.asarray(ovov) - numpy.asarray(eris.ovov)).max())

    emp2, v1, v2 = mycc.init_amps(eris)
    if rank == 0:
        print(lib.finger(v1) - 0.20852878109950079)
        print(lib.finger(v2) - 0.21333574169417541)
        print(emp2 - -0.12037888088751542)

    t1 = numpy.random.random((nocc,nvir))
    t2 = numpy.random.random((nocc,nocc,nvir,nvir))
    t2 = t2 + t2.transpose(1,0,3,2)
    v1, v2 = mycc.update_amps(t1, t2, eris)
    if rank == 0:
        print(lib.finger(v1) - 9.6029949445427079)
        print(lib.finger(v2) - 4.5308876217231813)
