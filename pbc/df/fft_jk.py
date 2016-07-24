#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
JK with discrete Fourier transformation
'''

import time
import numpy

from pyscf import lib
from pyscf.pbc import tools
from pyscf.pbc.df import fft_jk
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import numint

from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi

comm = mpi.comm
rank = mpi.rank


def get_j_kpts(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), kpt_band=None):
    master_args = (mydf._reg_keys, dm_kpts, hermi, kpts, kpt_band)
    worker_args = (mydf._reg_keys, None, hermi, kpts, kpt_band)
    return mpi.pool.apply(_get_j_kpts_wrap, master_args, worker_args)
def _get_j_kpts_wrap(args):
    from mpi4pyscf.pbc.df import fft_jk
    return fft_jk._get_j_kpts(*args)
def _get_j_kpts(reg_keys, dm_kpts, hermi=1,
                kpts=numpy.zeros((1,3)), kpt_band=None):
    mydf = _load_df(reg_keys)
    cell = mydf.cell
    gs = mydf.gs

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    coulG = tools.get_coulG(cell, gs=gs)
    ngs = len(coulG)

    rhoR = numpy.zeros((nset,ngs))
    for k, aoR in mydf.mpi_aoR_loop(cell, gs, kpts):
        for i in range(nset):
            rhoR[i] += numint.eval_rho(cell, aoR, dms[i,k])
    vR = rhoR = mpi.allreduce(rhoR)
    for i in range(nset):
        rhoR[i] *= 1./nkpts
        rhoG = tools.fft(rhoR[i], gs)
        vG = coulG * rhoG
        vR[i] = tools.ifft(vG, gs).real

    if kpt_band is not None:
        if rank == 0:
            for aoR_kband in mydf.aoR_loop(cell, gs, kpts, kpt_band):
                pass
            vj_kpts = [cell.vol/ngs * lib.dot(aoR_kband.T.conj()*vR[i], aoR_kband)
                       for i in range(nset)]
            if dm_kpts.ndim == 3:  # One set of dm_kpts for KRHF
                vj_kpts = vj_kpts[0]
            return lib.asarray(vj_kpts)
    else:
        vj_kpts = []
        weight = cell.vol / ngs
        for k, aoR in mydf.mpi_aoR_loop(cell, gs, kpts):
            for i in range(nset):
                vj_kpts.append(weight * lib.dot(aoR.T.conj()*vR[i], aoR))
        vj_kpts = mpi.gather(lib.asarray(vj_kpts, dtype=numpy.complex128))
        if rank == 0:
            vj_kpts = vj_kpts.reshape(nkpts,nset,nao,nao)
            return vj_kpts.transpose(1,0,2,3).reshape(dm_kpts.shape)

def get_k_kpts(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)), kpt_band=None):
    master_args = (mydf._reg_keys, dm_kpts, hermi, kpts, kpt_band)
    worker_args = (mydf._reg_keys, None, hermi, kpts, kpt_band)
    return mpi.pool.apply(_get_k_kpts_wrap, master_args, worker_args)
def _get_k_kpts_wrap(args):
    from mpi4pyscf.pbc.df import fft_jk
    return fft_jk._get_k_kpts(*args)
def _get_k_kpts(reg_keys, dm_kpts, hermi=1,
                kpts=numpy.zeros((1,3)), kpt_band=None):
    mydf = _load_df(reg_keys)
    cell = mydf.cell
    gs = mydf.gs
    coords = gen_grid.gen_uniform_grids(cell, gs)
    ngs = coords.shape[0]

    kpts = numpy.asarray(kpts)
    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    weight = 1./nkpts * (cell.vol/ngs)

    if kpt_band is not None:
        if abs(kpts).sum() < 1e-9 and abs(kpt_band).sum() < 1e-9:
            vk_kpts = numpy.zeros((nset,nao,nao), dtype=dms.dtype)
        else:
            vk_kpts = numpy.zeros((nset,nao,nao), dtype=numpy.complex128)
        for k, aoR_kband in mydf.aoR_loop(cell, gs, kpts, kpt_band):
            pass
        for k2, ao_k2 in mydf.mpi_aoR_loop(cell, gs, kpts):
            kpt2 = kpts[k2]
            vkR_k1k2 = fft_jk.get_vkR(mydf, cell, aoR_kband, ao_k2, kpt_band, kpt2, coords)
            #:vk_kpts = 1./nkpts * (cell.vol/ngs) * numpy.einsum('rs,Rp,Rqs,Rr->pq',
            #:            dm_kpts[k2], aoR_kband.conj(), vkR_k1k2, ao_k2)
            for i in range(nset):
                aoR_dm = lib.dot(ao_k2, dms[i,k2])
                tmp_Rq = numpy.einsum('Rqs,Rs->Rq', vkR_k1k2, aoR_dm)
                vk_kpts[i] += weight * lib.dot(aoR_kband.T.conj(), tmp_Rq)
            vkR_k1k2 = aoR_dm = tmp_Rq = None
        vk_kpts = mpi.reduce(lib.asarray(vk_kpts))
        if rank == 0:
            if dm_kpts.ndim == 3:
                vk_kpts = vk_kpts[0]
            return lib.asarray(vk_kpts)
    else:
        if abs(kpts).sum() < 1e-9:
            vk_kpts = numpy.zeros((nset,nkpts,nao,nao), dtype=dms.dtype)
        else:
            vk_kpts = numpy.zeros((nset,nkpts,nao,nao), dtype=numpy.complex128)
# TODO load balance for the nested loops
        for k2, ao_k2 in mydf.mpi_aoR_loop(cell, gs, kpts):
            kpt2 = kpts[k2]
            aoR_dms = [lib.dot(ao_k2, dms[i,k2]) for i in range(nset)]
            for k1, ao_k1 in mydf.aoR_loop(cell, gs, kpts):
                kpt1 = kpts[k1]
                vkR_k1k2 = fft_jk.get_vkR(mydf, cell, ao_k1, ao_k2, kpt1, kpt2, coords)
                for i in range(nset):
                    tmp_Rq = numpy.einsum('Rqs,Rs->Rq', vkR_k1k2, aoR_dms[i])
                    vk_kpts[i,k1] += weight * lib.dot(ao_k1.T.conj(), tmp_Rq)
            vkR_k1k2 = aoR_dms = tmp_Rq = None
        vk_kpts = mpi.reduce(lib.asarray(vk_kpts))
        if rank == 0:
            return vk_kpts.reshape(dm_kpts.shape)


def get_jk(mydf, dm, hermi=1, kpt=numpy.zeros(3), kpt_band=None):
    dm = numpy.asarray(dm, order='C')
    vj = get_j(mydf, dm, hermi, kpt, kpt_band)
    vk = get_k(mydf, dm, hermi, kpt, kpt_band)
    return vj, vk

def get_j(mydf, dm, hermi=1, kpt=numpy.zeros(3), kpt_band=None):
    dm = numpy.asarray(dm, order='C')
    nao = dm.shape[-1]
    dm_kpts = dm.reshape(-1,1,nao,nao)
    vj = get_j_kpts(mydf, dm_kpts, hermi, [kpt], kpt_band)
    return vj.reshape(dm.shape)

def get_k(mydf, dm, hermi=1, kpt=numpy.zeros(3), kpt_band=None):
    dm = numpy.asarray(dm, order='C')
    nao = dm.shape[-1]
    dm_kpts = dm.reshape(-1,1,nao,nao)
    vk = get_k_kpts(mydf, dm_kpts, hermi, [kpt], kpt_band)
    return vk.reshape(dm.shape)

def _load_df(reg_keys):
    mydf = mpi._registry[reg_keys[rank]]
    mydf.kpts, mydf.gs, mydf.exxdiv = \
            comm.bcast((mydf.kpts, mydf.gs, mydf.exxdiv))
    return mydf

def _format_dms(dm_kpts, kpts):
    if rank == 0:
        nkpts = len(kpts)
        nao = dm_kpts.shape[-1]
        dms = dm_kpts.reshape(-1,nkpts,nao,nao)
        comm.bcast((dms.shape, dms.dtype))
        comm.Bcast(dms)
    else:
        shape, dtype = comm.bcast()
        nao = shape[-1]
        dms = numpy.empty(shape, dtype=dtype)
        comm.Bcast(dms)
    return dms

