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
from pyscf.pbc.df.df_jk import is_zero, gamma_point
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import numint

from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi

comm = mpi.comm
rank = mpi.rank


@mpi.parallel_call
def get_j_kpts(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)),
               kpts_band=None):
    mydf = _sync_mydf(mydf)
    cell = mydf.cell
    gs = mydf.gs

    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    coulG = tools.get_coulG(cell, gs=gs)
    ngs = len(coulG)

    vR = rhoR = numpy.zeros((nset,ngs))
    for k, aoR in mydf.mpi_aoR_loop(gs, kpts):
        for i in range(nset):
            rhoR[i] += numint.eval_rho(cell, aoR, dms[i,k])
    vR = rhoR = mpi.allreduce(rhoR)
    for i in range(nset):
        rhoR[i] *= 1./nkpts
        rhoG = tools.fft(rhoR[i], gs)
        vG = coulG * rhoG
        vR[i] = tools.ifft(vG, gs).real

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    weight = cell.vol / ngs
    vj_kpts = numpy.zeros((nset,nband,nao,nao), dtype=numpy.complex128)
    for k, aoR in mydf.mpi_aoR_loop(gs, kpts_band):
        for i in range(nset):
            vj_kpts[i,k] = weight * lib.dot(aoR.T.conj()*vR[i], aoR)

    vj_kpts = mpi.reduce(lib.asarray(vj_kpts))
    if gamma_point(kpts_band):
        vj_kpts = vj_kpts.real
    return _format_jks(vj_kpts, dm_kpts, input_band, kpts)

@mpi.parallel_call
def get_k_kpts(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)),
               kpts_band=None, exxdiv=None):
    mydf = _sync_mydf(mydf)
    cell = mydf.cell
    gs = mydf.gs
    coords = cell.gen_uniform_grids(gs)
    ngs = coords.shape[0]

    kpts = numpy.asarray(kpts)
    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    weight = 1./nkpts * (cell.vol/ngs)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    vk_kpts = numpy.zeros((nset,nband,nao,nao), dtype=numpy.complex128)

    max_memory = mydf.max_memory - lib.current_memory()[0]
    blksize = int(max(max_memory*1e6/16/2/ngs/nao, 1))
    vR_dm = numpy.empty((nset,nao,ngs), dtype=numpy.complex128)
    for k2, ao_k2 in mydf.mpi_aoR_loop(gs, kpts):
        kpt2 = kpts[k2]
        ao_dms = [lib.dot(dms[i,k2], ao_k2.conj().T) for i in range(nset)]

        for k1, ao_k1 in mydf.aoR_loop(gs, kpts_band):
            kpt1 = kpts_band[k1]
            mydf.exxdiv = exxdiv
            coulG = tools.get_coulG(cell, kpt2-kpt1, True, mydf, gs)
            if is_zero(kpt1-kpt2):
                expmikr = numpy.array(1.)
            else:
                expmikr = numpy.exp(-1j * numpy.dot(coords, kpt2-kpt1))[:,None]

            for p0, p1 in lib.prange(0, nao, blksize):
                rho1 = numpy.einsum('gi,gj->ijg', ao_k1[:,p0:p1].conj()*expmikr, ao_k2)
                vG = tools.fft(rho1.reshape(-1,ngs), gs)
                vG *= coulG
                vR = tools.ifft(vG, gs).reshape(p1-p0,nao,ngs)
                vG = None
                if vR_dm.dtype == numpy.double:
                    vR = vR.real
                for i in range(nset):
                    vR_dm[i,p0:p1] = numpy.einsum('ijg,jg->ig', vR, ao_dms[i])
                vR = None
            vR_dm *= expmikr.conj()

            for i in range(nset):
                vk_kpts[i,k1] += weight * lib.dot(vR_dm[i], ao_k1)

    vk_kpts = mpi.reduce(lib.asarray(vk_kpts))
    if gamma_point(kpts_band) and gamma_point(kpts):
        vk_kpts = vk_kpts.real
    return _format_jks(vk_kpts, dm_kpts, input_band, kpts)


def get_jk(mydf, dm, hermi=1, kpt=numpy.zeros(3), kpt_band=None,
           with_j=True, with_k=True, exxdiv=None):
    dm = numpy.asarray(dm, order='C')
    vj = vk = None
    if with_j:
        vj = get_j(mydf, dm, hermi, kpt, kpt_band)
    if with_k:
        vk = get_k(mydf, dm, hermi, kpt, kpt_band, exxdiv)
    return vj, vk

def get_j(mydf, dm, hermi=1, kpt=numpy.zeros(3), kpt_band=None):
    dm = numpy.asarray(dm, order='C')
    nao = dm.shape[-1]
    dm_kpts = dm.reshape(-1,1,nao,nao)
    vj = get_j_kpts(mydf, dm_kpts, hermi, kpt.reshape(1,3), kpt_band)
    return vj.reshape(dm.shape)

def get_k(mydf, dm, hermi=1, kpt=numpy.zeros(3), kpt_band=None, exxdiv=None):
    dm = numpy.asarray(dm, order='C')
    nao = dm.shape[-1]
    dm_kpts = dm.reshape(-1,1,nao,nao)
    vk = get_k_kpts(mydf, dm_kpts, hermi, kpt.reshape(1,3), kpt_band, exxdiv)
    return vk.reshape(dm.shape)

def _sync_mydf(mydf):
    mydf.unpack_(comm.bcast(mydf.pack()))
    return mydf

