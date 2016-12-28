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
import numpy

from pyscf import lib
from pyscf.pbc import tools
from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0
from pyscf.pbc.df.df_jk import zdotNN, zdotNC, zdotCN

from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi
from mpi4pyscf.pbc.df.pwdf_jk import _format_dms

comm = mpi.comm
rank = mpi.rank

#
# Split the Coulomb potential to two parts.  Computing short range part in
# real space, long range part in reciprocal space.
#

def density_fit(mf, auxbasis=None, gs=None, with_df=None):
    '''Generte density-fitting SCF object

    Args:
        auxbasis : str or basis dict
            Same format to the input attribute mol.basis.
            The default basis 'weigend+etb' means weigend-coulomb-fit basis
            for light elements and even-tempered basis for heavy elements.
        gs : tuple
            number of grids in each (+)direction
        with_df : MDF object
    '''
    from mpi4pyscf.pbc.df import mdf
    if with_df is None:
        if hasattr(mf, 'kpts'):
            kpts = mf.kpts
        else:
            kpts = numpy.reshape(mf.kpt, (1,3))
        with_df = mdf.MDF(mf.cell, kpts)
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose
        with_df.auxbasis = auxbasis
        if gs is not None:
            with_df.gs = gs

    mf = copy.copy(mf)
    mf.with_df = with_df
    return mf


def _get_j_kpts(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)),
                kpt_band=None):
    mydf = _sync_mydf(mydf)
    cell = mydf.cell
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = (time.clock(), time.time())

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    naux = mydf.auxcell.nao_nr()

    if kpt_band is None:
        kpts_band = kpts
    else:
        kpts_band = numpy.reshape(kpt_band, (-1,3))
    nband = len(kpts_band)
    j_real = gamma_point(kpts_band)

    dmsR = dms.real.reshape(nset,nkpts,nao**2)
    dmsI = dms.imag.reshape(nset,nkpts,nao**2)
    kpt_allow = numpy.zeros(3)
    coulG = mydf.weighted_coulG(kpt_allow, False, mydf.gs)
    ngs = len(coulG)
    vR = numpy.zeros((nset,ngs))
    vI = numpy.zeros((nset,ngs))
    max_memory = max(2000, (mydf.max_memory - lib.current_memory()[0]) * .9)
    for k, pqkR, pqkI, p0, p1 \
            in mydf.ft_loop(mydf.gs, kpt_allow, kpts, max_memory=max_memory):
        # contract dm to rho_rs(-G+k_rs)  (Note no .T on dm)
        # rho_rs(-G+k_rs) is computed as conj(rho_{rs^*}(G-k_rs))
        #               == conj(transpose(rho_sr(G+k_sr), (0,2,1)))
        for i in range(nset):
            rhoR = numpy.dot(dmsR[i,k], pqkR)
            rhoR+= numpy.dot(dmsI[i,k], pqkI)
            rhoI = numpy.dot(dmsI[i,k], pqkR)
            rhoI-= numpy.dot(dmsR[i,k], pqkI)
            vR[i,p0:p1] += rhoR * coulG[p0:p1]
            vI[i,p0:p1] += rhoI * coulG[p0:p1]
        pqkR = pqkI = None
    weight = 1./nkpts
    vR *= weight
    vI *= weight
    vR = mpi.allreduce(vR)
    vI = mpi.allreduce(vI)
    t1 = log.timer_debug1('get_j pass 1 to compute J(G)', *t1)

    if kpt_band is None:
        kpts_band = kpts
    else:
        kpts_band = numpy.reshape(kpt_band, (-1,3))

    vjR = numpy.zeros((nset,nband,nao*nao))
    vjI = numpy.zeros((nset,nband,nao*nao))
    for k, pqkR, pqkI, p0, p1 \
            in mydf.ft_loop(mydf.gs, kpt_allow, kpts_band, max_memory=max_memory):
        for i in range(nset):
            vjR[i,k] += numpy.dot(pqkR, vR[i,p0:p1])
            vjR[i,k] -= numpy.dot(pqkI, vI[i,p0:p1])
            if not j_real:
                vjI[i,k] += numpy.dot(pqkI, vR[i,p0:p1])
                vjI[i,k] += numpy.dot(pqkR, vI[i,p0:p1])
        pqkR = pqkI = None

    rhoR  = numpy.zeros((nset,naux))
    rhoI  = numpy.zeros((nset,naux))
    jauxR = numpy.zeros((nset,naux))
    jauxI = numpy.zeros((nset,naux))
    for k, kpt in enumerate(kpts_band):
        kptii = numpy.asarray((kpt,kpt))
        p1 = 0
        for LpqR, LpqI, j3cR, j3cI in mydf.sr_loop(kptii, max_memory, False):
            p0, p1 = p1, p1+LpqR.shape[0]
            #:Lpq = (LpqR + LpqI*1j).transpose(1,0,2)
            #:j3c = (j3cR + j3cI*1j).transpose(1,0,2)
            #:rho [:,p0:p1] += numpy.einsum('Lpq,xpq->xL', Lpq, dms[:,k])
            #:jaux[:,p0:p1] += numpy.einsum('Lpq,xpq->xL', j3c, dms[:,k])
            rhoR [:,p0:p1]+= numpy.einsum('Lp,xp->xL', LpqR, dmsR[:,k])
            rhoR [:,p0:p1]-= numpy.einsum('Lp,xp->xL', LpqI, dmsI[:,k])
            rhoI [:,p0:p1]+= numpy.einsum('Lp,xp->xL', LpqR, dmsI[:,k])
            rhoI [:,p0:p1]+= numpy.einsum('Lp,xp->xL', LpqI, dmsR[:,k])
            jauxR[:,p0:p1]+= numpy.einsum('Lp,xp->xL', j3cR, dmsR[:,k])
            jauxR[:,p0:p1]-= numpy.einsum('Lp,xp->xL', j3cI, dmsI[:,k])
            jauxI[:,p0:p1]+= numpy.einsum('Lp,xp->xL', j3cR, dmsI[:,k])
            jauxI[:,p0:p1]+= numpy.einsum('Lp,xp->xL', j3cI, dmsR[:,k])
            LpqR = LpqI = j3cR = j3cI = None

    weight = 1./nkpts
    jauxR *= weight
    jauxI *= weight
    rhoR  *= weight
    rhoI  *= weight
    vjR = vjR.reshape(nset,nband,nao,nao)
    vjI = vjI.reshape(nset,nband,nao,nao)
    for k, kpt in enumerate(kpts_band):
        kptii = numpy.asarray((kpt,kpt))
        p1 = 0
        for LpqR, LpqI, j3cR, j3cI in mydf.sr_loop(kptii, max_memory, True):
            p0, p1 = p1, p1+LpqR.shape[0]
            #:v = numpy.dot(jaux, Lpq) + numpy.dot(rho, j3c)
            #:vj_kpts[:,k] += lib.unpack_tril(v)
            v  = numpy.dot(jauxR[:,p0:p1], LpqR)
            v -= numpy.dot(jauxI[:,p0:p1], LpqI)
            v += numpy.dot(rhoR [:,p0:p1], j3cR)
            v -= numpy.dot(rhoI [:,p0:p1], j3cI)
            vjR[:,k] += lib.unpack_tril(v)
            if not j_real:
                v  = numpy.dot(jauxR[:,p0:p1], LpqI)
                v += numpy.dot(jauxI[:,p0:p1], LpqR)
                v += numpy.dot(rhoR [:,p0:p1], j3cI)
                v += numpy.dot(rhoI [:,p0:p1], j3cR)
                vjI[:,k] += lib.unpack_tril(v, lib.ANTIHERMI)
            LpqR = LpqI = j3cR = j3cI = None
    t1 = log.timer_debug1('get_j pass 2', *t1)

    vjR = mpi.reduce(vjR)
    vjI = mpi.reduce(vjI)
    if rank == 0:
        if j_real:
            vj_kpts = vjR
        else:
            vj_kpts = vjR + vjI*1j

        if kpt_band is not None and numpy.shape(kpt_band) == (3,):
            if nset == 1:  # One set of dm_kpts for KRHF
                return vj_kpts[0,0]
            else:
                return vj_kpts[:,0]
        else:
            return vj_kpts.reshape(dm_kpts.shape)
get_j_kpts = mpi.parallel_call(_get_j_kpts)


def _get_k_kpts(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)),
                kpt_band=None, exxdiv=None):
    mydf = _sync_mydf(mydf)
    cell = mydf.cell
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = (time.clock(), time.time())

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    naux = mydf.auxcell.nao_nr()
    nao_pair = nao * (nao+1) // 2

    if kpt_band is None:
        kpts_band = kpts
        swap_2e = True
    else:
        kpts_band = numpy.reshape(kpt_band, (-1,3))
    nband = len(kpts_band)
    kk_table = kpts_band.reshape(-1,1,3) - kpts.reshape(1,-1,3)
    kk_todo = numpy.ones(kk_table.shape[:2], dtype=bool)
    vkR = numpy.zeros((nset,nband,nao,nao))
    vkI = numpy.zeros((nset,nband,nao,nao))
    dmsR = numpy.asarray(dms.real, order='C')
    dmsI = numpy.asarray(dms.imag, order='C')

    # K_pq = ( p{k1} i{k2} | i{k2} q{k1} )
    def make_kpt(kpt):  # kpt = kptj - kpti
        # search for all possible ki and kj that has ki-kj+kpt=0
        kk_match = numpy.einsum('ijx->ij', abs(kk_table + kpt)) < 1e-9
        kpti_idx, kptj_idx = numpy.where(kk_todo & kk_match)
        nkptj = len(kptj_idx)
        log.alldebug1('kptj - kpti = %s', kpt)
        log.debug2('kpti_idx = %s', kpti_idx)
        log.debug2('kptj_idx = %s', kptj_idx)
        kk_todo[kpti_idx,kptj_idx] = False
        if swap_2e and not is_zero(kpt):
            kk_todo[kptj_idx,kpti_idx] = False

        max_memory = max(2000, (mydf.max_memory-lib.current_memory()[0]))*.9
        max_memory = max_memory * (nkptj+1)/(nkptj+5)
        blksize = max(int(max_memory*4e6/(nkptj+5)/16/nao**2), 16)
        bufR = numpy.empty((blksize*nao**2))
        bufI = numpy.empty((blksize*nao**2))
        mydf.exxdiv = exxdiv
        vkcoulG = mydf.weighted_coulG(kpt, True, mydf.gs)
        kptjs = kpts[kptj_idx]
        # <r|-G+k_rs|s> = conj(<s|G-k_rs|r>) = conj(<s|G+k_sr|r>)
        for k, pqkR, pqkI, p0, p1 \
                in mydf.ft_loop(mydf.gs, kpt, kptjs, max_memory=max_memory):
            ki = kpti_idx[k]
            kj = kptj_idx[k]
            coulG = numpy.sqrt(vkcoulG[p0:p1])

# case 1: k_pq = (pi|iq)
#:v4 = numpy.einsum('ijL,lkL->ijkl', pqk, pqk.conj())
#:vk += numpy.einsum('ijkl,jk->il', v4, dm)
            pqkR *= coulG
            pqkI *= coulG
            pLqR = lib.transpose(pqkR.reshape(nao,nao,-1), axes=(0,2,1), out=bufR)
            pLqI = lib.transpose(pqkI.reshape(nao,nao,-1), axes=(0,2,1), out=bufI)
            iLkR = numpy.empty((nao*(p1-p0),nao))
            iLkI = numpy.empty((nao*(p1-p0),nao))
            for i in range(nset):
                iLkR, iLkI = zdotNN(pLqR.reshape(-1,nao), pLqI.reshape(-1,nao),
                                    dmsR[i,kj], dmsI[i,kj], 1, iLkR, iLkI)
                zdotNC(iLkR.reshape(nao,-1), iLkI.reshape(nao,-1),
                       pLqR.reshape(nao,-1).T, pLqI.reshape(nao,-1).T,
                       1, vkR[i,ki], vkI[i,ki], 1)

# case 2: k_pq = (iq|pi)
#:v4 = numpy.einsum('iLj,lLk->ijkl', pqk, pqk.conj())
#:vk += numpy.einsum('ijkl,li->kj', v4, dm)
            if swap_2e and not is_zero(kpt):
                iLkR = iLkR.reshape(nao,-1)
                iLkI = iLkI.reshape(nao,-1)
                for i in range(nset):
                    iLkR, iLkI = zdotNN(dmsR[i,ki], dmsI[i,ki], pLqR.reshape(nao,-1),
                                        pLqI.reshape(nao,-1), 1, iLkR, iLkI)
                    zdotCN(pLqR.reshape(-1,nao).T, pLqI.reshape(-1,nao).T,
                           iLkR.reshape(-1,nao), iLkI.reshape(-1,nao),
                           1, vkR[i,kj], vkI[i,kj], 1)
            pqkR = pqkI = coulG = pLqR = pLqI = iLkR = iLkI = None

        # Note: kj-ki for electorn 1 and ki-kj for electron 2
        # j2c ~ ({kj-ki}|{ks-kr}) ~ ({kj-ki}|-{kj-ki}) ~ ({kj-ki}|{ki-kj})
        # j3c ~ (Q|kj,ki) = j3c{ji} = (Q|ki,kj)* = conj(transpose(j3c{ij}, (0,2,1)))

        bufR = numpy.empty((mydf.blockdim*nao**2))
        bufI = numpy.empty((mydf.blockdim*nao**2))
        max_memory = max(2000, mydf.max_memory-lib.current_memory()[0])
        for ki,kj in zip(kpti_idx,kptj_idx):
            kpti = kpts_band[ki]
            kptj = kpts[kj]
            kptij = numpy.asarray((kpti,kptj))
            for LpqR, LpqI, j3cR, j3cI in mydf.sr_loop(kptij, max_memory, False):
                nrow = LpqR.shape[0]
                pLqR = numpy.ndarray((nao,nrow,nao), buffer=bufR)
                pLqI = numpy.ndarray((nao,nrow,nao), buffer=bufI)
                pjqR = numpy.ndarray((nao,nrow,nao), buffer=LpqR)
                pjqI = numpy.ndarray((nao,nrow,nao), buffer=LpqI)
                tmpR = numpy.ndarray((nao,nrow*nao), buffer=j3cR)
                tmpI = numpy.ndarray((nao,nrow*nao), buffer=j3cI)
                pLqR[:] = LpqR.reshape(-1,nao,nao).transpose(1,0,2)
                pLqI[:] = LpqI.reshape(-1,nao,nao).transpose(1,0,2)
                pjqR[:] = j3cR.reshape(-1,nao,nao).transpose(1,0,2)
                pjqI[:] = j3cI.reshape(-1,nao,nao).transpose(1,0,2)
                # K ~ 'iLj,lLk*,li->kj' + 'lLk*,iLj,li->kj'
                for i in range(nset):
                    tmpR, tmpI = zdotNN(dmsR[i,ki], dmsI[i,ki], pjqR.reshape(nao,-1),
                                        pjqI.reshape(nao,-1), 1, tmpR, tmpI)
                    vk1R, vk1I = zdotCN(pLqR.reshape(-1,nao).T, pLqI.reshape(-1,nao).T,
                                        tmpR.reshape(-1,nao), tmpI.reshape(-1,nao))
                    vkR[i,kj] += vk1R
                    vkI[i,kj] += vk1I
                    if hermi:
                        vkR[i,kj] += vk1R.T
                        vkI[i,kj] -= vk1I.T
                    else:
                        tmpR, tmpI = zdotNN(dmsR[i,ki], dmsI[i,ki], pLqR.reshape(nao,-1),
                                            pLqI.reshape(nao,-1), 1, tmpR, tmpI)
                        zdotCN(pjqR.reshape(-1,nao).T, pjqI.reshape(-1,nao).T,
                               tmpR.reshape(-1,nao), tmpI.reshape(-1,nao),
                               1, vkR[i,kj], vkI[i,kj], 1)

                if swap_2e and not is_zero(kpt):
                    tmpR = tmpR.reshape(nao*nrow,nao)
                    tmpI = tmpI.reshape(nao*nrow,nao)
                    # K ~ 'iLj,lLk*,jk->il' + 'lLk*,iLj,jk->il'
                    for i in range(nset):
                        tmpR, tmpI = zdotNN(pjqR.reshape(-1,nao), pjqI.reshape(-1,nao),
                                            dmsR[i,kj], dmsI[i,kj], 1, tmpR, tmpI)
                        vk1R, vk1I = zdotNC(tmpR.reshape(nao,-1), tmpI.reshape(nao,-1),
                                            pLqR.reshape(nao,-1).T, pLqI.reshape(nao,-1).T)
                        vkR[i,ki] += vk1R
                        vkI[i,ki] += vk1I
                        if hermi:
                            vkR[i,ki] += vk1R.T
                            vkI[i,ki] -= vk1I.T
                        else:
                            tmpR, tmpI = zdotNN(pLqR.reshape(-1,nao), pLqI.reshape(-1,nao),
                                                dmsR[i,kj], dmsI[i,kj], 1, tmpR, tmpI)
                            zdotNC(tmpR.reshape(nao,-1), tmpI.reshape(nao,-1),
                                   pjqR.reshape(nao,-1).T, pjqI.reshape(nao,-1).T,
                                   1, vkR[i,ki], vkI[i,ki], 1)
                LpqR = LpqI = j3cR = j3cI = tmpR = tmpI = None
        return None

    for ki, kpti in enumerate(kpts_band):
        for kj, kptj in enumerate(kpts):
            if kk_todo[ki,kj]:
                make_kpt(kptj-kpti)

    vkR = mpi.reduce(vkR)
    vkI = mpi.reduce(vkI)
    if rank == 0:
        if (gamma_point(kpts) and gamma_point(kpts_band) and
            not numpy.iscomplexobj(dm_kpts)):
            vk_kpts = vkR
        else:
            vk_kpts = vkR + vkI * 1j
        vk_kpts *= 1./nkpts
        t1 = log.timer_debug1('get_k', *t1)

        if kpt_band is not None and numpy.shape(kpt_band) == (3,):
            if nset == 1:  # One set of dm_kpts for KRHF
                return vk_kpts[0,0]
            else:
                return vk_kpts[:,0]
        else:
            return vk_kpts.reshape(dm_kpts.shape)
get_k_kpts = mpi.parallel_call(_get_k_kpts)


##################################################
#
# Single k-point
#
##################################################

@mpi.parallel_call
def get_jk(mydf, dm, hermi=1, kpt=numpy.zeros(3),
           kpt_band=None, with_j=True, with_k=True, exxdiv=None):
    '''JK for given k-point'''
    if mydf._cderi is None:
        mydf._build()

    vj = vk = None
    if kpt_band is not None and abs(kpt-kpt_band).sum() > 1e-9:
        kpt = numpy.reshape(kpt, (1,3))
        if with_k:
            vk = _get_k_kpts(mydf, [dm], hermi, kpt, kpt_band, exxdiv)
        if with_j:
            vj = _get_j_kpts(mydf, [dm], hermi, kpt, kpt_band)
        return vj, vk

    mydf = _sync_mydf(mydf)
    cell = mydf.cell
    log = logger.Logger(mydf.stdout, mydf.verbose)
    t0 = t1 = (time.clock(), time.time())

    dm = numpy.asarray(dm, order='C')
    dms = _format_dms(dm, [kpt])
    nset, _, nao = dms.shape[:3]
    dms = dms.reshape(nset,nao,nao)
    j_real = gamma_point(kpt)
    k_real = gamma_point(kpt) and not numpy.iscomplexobj(dms)

    naux = mydf.auxcell.nao_nr()
    kptii = numpy.asarray((kpt,kpt))
    kpt_allow = numpy.zeros(3)
    dmsR = dms.real.reshape(nset,nao,nao)
    dmsI = dms.imag.reshape(nset,nao,nao)
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, (mydf.max_memory - mem_now))
    if with_j:
        vjR = numpy.zeros((nset,nao,nao))
        vjI = numpy.zeros((nset,nao,nao))
    if with_k:
        vkR = numpy.zeros((nset,nao,nao))
        vkI = numpy.zeros((nset,nao,nao))
        buf1R = numpy.empty((mydf.blockdim*nao**2))
        buf2R = numpy.empty((mydf.blockdim*nao**2))
        buf1I = numpy.zeros((mydf.blockdim*nao**2))
        buf2I = numpy.empty((mydf.blockdim*nao**2))
        max_memory *= .5
    log.alldebug1('max_memory = %d MB (%d in use)', max_memory, mem_now)
    def contract_k(pLqR, pLqI):
        # K ~ 'iLj,lLk*,li->kj' + 'lLk*,iLj,li->kj'
        nrow = pLqR.shape[1]
        tmpR = numpy.ndarray((nao,nrow*nao), buffer=buf2R)
        if k_real:
            for i in range(nset):
                lib.ddot(dmsR[i], pLqR.reshape(nao,-1), 1, tmpR)
                lib.ddot(pLqR.reshape(-1,nao).T, tmpR.reshape(-1,nao), 1, vkR[i], 1)
        else:
            tmpI = numpy.ndarray((nao,nrow*nao), buffer=buf2I)
            for i in range(nset):
                zdotNN(dmsR[i], dmsI[i], pLqR.reshape(nao,-1),
                       pLqI.reshape(nao,-1), 1, tmpR, tmpI, 0)
                zdotCN(pLqR.reshape(-1,nao).T, pLqI.reshape(-1,nao).T,
                       tmpR.reshape(-1,nao), tmpI.reshape(-1,nao),
                       1, vkR[i], vkI[i], 1)
    pLqI = None
    thread_k = None
    for LpqR, LpqI in mydf.sr_loop(kptii, max_memory, False):
        LpqR = LpqR.reshape(-1,nao,nao)
        t1 = log.alltimer_debug2('        load', *t1)
        if thread_k is not None:
            thread_k.join()
        if with_j:
            rhoR  = numpy.einsum('Lpq,xpq->xL', LpqR, dmsR)
            if not j_real:
                LpqI = LpqI.reshape(-1,nao,nao)
                rhoR -= numpy.einsum('Lpq,xpq->xL', LpqI, dmsI)
                rhoI  = numpy.einsum('Lpq,xpq->xL', LpqR, dmsI)
                rhoI += numpy.einsum('Lpq,xpq->xL', LpqI, dmsR)
            vjR += numpy.einsum('xL,Lpq->xpq', rhoR, LpqR)
            if not j_real:
                vjR -= numpy.einsum('xL,Lpq->xpq', rhoI, LpqI)
                vjI += numpy.einsum('xL,Lpq->xpq', rhoR, LpqI)
                vjI += numpy.einsum('xL,Lpq->xpq', rhoI, LpqR)

        t1 = log.alltimer_debug2('        with_j', *t1)
        if with_k:
            nrow = LpqR.shape[0]
            pLqR = numpy.ndarray((nao,nrow,nao), buffer=buf1R)
            pLqR[:] = LpqR.transpose(1,0,2)
            if not k_real:
                pLqI = numpy.ndarray((nao,nrow,nao), buffer=buf1I)
                if LpqI is not None:
                    pLqI[:] = LpqI.reshape(-1,nao,nao).transpose(1,0,2)

            thread_k = lib.background_thread(contract_k, pLqR, pLqI)
            t1 = log.alltimer_debug2('        with_k', *t1)
        LpqR = LpqI = pLqR = pLqI = tmpR = tmpI = None
    if thread_k is not None:
        thread_k.join()
    thread_k = None

    if with_j:
        vjR = mpi.reduce(vjR)
        vjI = mpi.reduce(vjI)
    if with_k:
        vkR = mpi.reduce(vkR)
        vkI = mpi.reduce(vkI)
    t0 = log.alltimer_debug2('get_jk', *t0)

    if rank == 0:
        if with_j:
            if j_real:
                vj = vjR
            else:
                vj = vjR + vjI * 1j
            vj = vj.reshape(dm.shape)
        if with_k:
            if k_real:
                vk = vkR
            else:
                vk = vkR + vkI * 1j
            if exxdiv is not None:
                assert(exxdiv.lower() == 'ewald')
                _ewald_exxdiv_for_G0(cell, kpt, dms, vk)
            vk = vk.reshape(dm.shape)
        print abs(vj).sum(), abs(vk).sum(), 'mpi'
        t1 = log.timer_debug1('sr jk', *t1)
        return vj, vk


def is_zero(kpt):
    return kpt is None or abs(kpt).sum() < 1e-9
gamma_point = is_zero

def _sync_mydf(mydf):
    return mydf.unpack_(comm.bcast(mydf.pack()))


if __name__ == '__main__':
    import pyscf.pbc.gto as pgto
    import pyscf.pbc.scf as pscf
    import pyscf.pbc.dft as pdft

    L = 5.
    n = 5
    cell = pgto.Cell()
    cell.a = numpy.diag([L,L,L])
    cell.gs = numpy.array([n,n,n])

    cell.atom = '''C    3.    2.       3.
                   C    1.    1.       1.'''
    #cell.basis = {'He': [[0, (1.0, 1.0)]]}
    #cell.basis = '631g'
    #cell.basis = {'He': [[0, (2.4, 1)], [1, (1.1, 1)]]}
    cell.basis = 'ccpvdz'
    cell.verbose = 0
    cell.build(0,0)
    cell.verbose = 5

    mf = pscf.RHF(cell)
    auxbasis = 'weigend'
    mf = density_fit(mf, auxbasis)
    mf.with_df.gs = (5,) * 3
    mf.with_df.approx_sr_level = 3
    dm = mf.get_init_guess()
    vj = mf.get_j(cell, dm)
    print(numpy.einsum('ij,ji->', vj, dm), 'ref=46.69745030912447')
    vj, vk = mf.get_jk(cell, dm)
    print(numpy.einsum('ij,ji->', vj, dm), 'ref=46.69745030912447')
    print(numpy.einsum('ij,ji->', vk, dm), 'ref=37.33704732444835')
    print(numpy.einsum('ij,ji->', mf.get_hcore(cell), dm), 'ref=-75.574414055823766')
