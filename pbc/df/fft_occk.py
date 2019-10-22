#!/usr/bin/env python
#
# Author: Peng Bao <baopeng@iccas.ac.cn>
#
# Modified from mpi4pyscf/pbc/df/fft_jk.py
#
# ref: Lin, L. J. Chem. Theory Comput. 2016, 12, 2242\u22122249.
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
from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import numint

import scipy.linalg

from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi

comm = mpi.comm
rank = mpi.rank

@mpi.parallel_call
def get_k_kpts_occ(mydf, dm_kpts, hermi=1, kpts=numpy.zeros((1,3)),
               kpts_band=None, exxdiv=None):

    mydf = _sync_mydf(mydf)
    cell = mydf.cell
    mesh = mydf.mesh
    coords = cell.gen_uniform_grids(mesh)
    ngrids = coords.shape[0]

    if hasattr(dm_kpts, 'mo_coeff'):
        if dm_kpts.ndim == 3:  # KRHF
            mo_coeff = [dm_kpts.mo_coeff]
            mo_occ   = [dm_kpts.mo_occ  ]
        else:  # KUHF
            mo_coeff = dm_kpts.mo_coeff
            mo_occ   = dm_kpts.mo_occ
    elif hasattr(dm_kpts[0], 'mo_coeff'):
        mo_coeff = [dm.mo_coeff for dm in dm_kpts]
        mo_occ   = [dm.mo_occ   for dm in dm_kpts]
    else:
        mo_coeff = None

    kpts = numpy.asarray(kpts)
    dm_kpts = lib.asarray(dm_kpts, order='C')
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]

    weight = 1./nkpts * (cell.vol/ngrids)

    kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)

    if gamma_point(kpts_band) and gamma_point(kpts):
        vk_kpts = numpy.zeros((nset,nband,nao,nao), dtype=dms.dtype)
    else:
        vk_kpts = numpy.zeros((nset,nband,nao,nao), dtype=numpy.complex128)

    coords = mydf.grids.coords
    ao2_kpts = [numpy.asarray(ao.T, order='C')
                for ao in mydf._numint.eval_ao(cell, coords, kpts=kpts)]
    if input_band is None:
        ao1_kpts = ao2_kpts
    else:
        ao1_kpts = [numpy.asarray(ao.T, order='C')
                    for ao in mydf._numint.eval_ao(cell, coords, kpts=kpts_band)]
    # occ
    if mo_coeff is None:
            return _format_jks(vk_kpts, dm_kpts, input_band, kpts)
    # occ
    if gamma_point(kpts_band) and gamma_point(kpts):
        for i in range(nset):
            occ = mo_occ[i][0]
            kiv = numpy.zeros((nset,nband,mo_coeff[i][0][:,occ>0].shape[1],nao), dtype=dms.dtype)
    else:
        kiv = [[]*nset]
        for i in range(nset):
            for k1 in range(nband):
                occ = mo_occ[i][k1]
                kiv[i].append(numpy.zeros((mo_coeff[i][k1][:,occ>0].shape[1],nao), dtype=numpy.complex128))

    mem_now = lib.current_memory()[0]
    max_memory = mydf.max_memory - mem_now
    blksize = int(min(nao, max(1, (max_memory-mem_now)*1e6/16/4/ngrids/nao)))
    lib.logger.debug1(mydf, 'max_memory %s  blksize %d', max_memory, blksize)
    ao1_dtype = numpy.result_type(*ao1_kpts)
    ao2_dtype = numpy.result_type(*ao2_kpts)
    vR_dm = numpy.empty((nset,nao,ngrids), dtype=vk_kpts.dtype)

    ao_dms_buf = [None] * nkpts
    # occ
    ao3T_buf = [None] * nkpts

    tasks = [(k1,k2) for k2 in range(nkpts) for k1 in range(nband)]
    for k1, k2 in mpi.static_partition(tasks):
        ao1T = ao1_kpts[k1]
        ao2T = ao2_kpts[k2]
        kpt1 = kpts_band[k1]
        kpt2 = kpts[k2]
        if ao2T.size == 0 or ao1T.size == 0:
            continue

        # If we have an ewald exxdiv, we add the G=0 correction near the
        # end of the function to bypass any discretization errors
        # that arise from the FFT.
        mydf.exxdiv = exxdiv
        if exxdiv == 'ewald' or exxdiv is None:
            coulG = tools.get_coulG(cell, kpt2-kpt1, False, mydf, mesh)
        else:
            coulG = tools.get_coulG(cell, kpt2-kpt1, True, mydf, mesh)
        if is_zero(kpt1-kpt2):
            expmikr = numpy.array(1.)
        else:
            expmikr = numpy.exp(-1j * numpy.dot(coords, kpt2-kpt1))

        if ao_dms_buf[k2] is None:
            if mo_coeff is None:
                ao_dms = [lib.dot(dm[k2], ao2T.conj()) for dm in dms]
            else:
                ao_dms = []
                for i, dm in enumerate(dms):
                    occ = mo_occ[i][k2]
                    mo_scaled = mo_coeff[i][k2][:,occ>0] * numpy.sqrt(occ[occ>0])
                    ao_dms.append(lib.dot(mo_scaled.T, ao2T).conj())
            ao_dms_buf[k2] = ao_dms
        else:
            ao_dms = ao_dms_buf[k2]

        if ao3T_buf[k1] is None:
            if mo_coeff is not None:           
                ao3T = []
                for i, dm in enumerate(dms):
                    occ = mo_occ[i][k1]
                    mo_scaled = mo_coeff[i][k1][:,occ>0] 
                    ao3T.append(lib.dot(mo_scaled.T, ao1T))
            ao3T_buf[k1] = ao3T
        else:
            ao3T = ao3T_buf[k1]

        if mo_coeff is None:
            for p0, p1 in lib.prange(0, nao, blksize):
                rho1 = numpy.einsum('ig,jg->ijg', ao1T[p0:p1].conj()*expmikr, ao2T)
                vG = tools.fft(rho1.reshape(-1,ngrids), mesh)
                rho1 = None
                vG *= coulG
                vR = tools.ifft(vG, mesh).reshape(p1-p0,nao,ngrids)
                vG = None
                if vR_dm.dtype == numpy.double:
                    vR = vR.real
                for i in range(nset):
                    numpy.einsum('ijg,jg->ig', vR, ao_dms[i], out=vR_dm[i,p0:p1])
                vR = None
        else:
            for i in range(nset):
                for p0, p1 in lib.prange(0, ao3T[i].shape[0], blksize):

                    rho1 = numpy.einsum('ig,jg->ijg',
                                        ao3T[i][p0:p1].conj()*expmikr,
                                        ao_dms[i].conj())
                    vG = tools.fft(rho1.reshape(-1,ngrids), mesh)
                    rho1 = None
                    vG *= coulG
                    vR = tools.ifft(vG, mesh).reshape(p1-p0,-1,ngrids)
                    vG = None
                    if vR_dm.dtype == numpy.double:
                        vR = vR.real
                    numpy.einsum('ijg,jg->ig', vR, ao_dms[i], out=vR_dm[i,p0:p1])
                    vR = None
        vR_dm *= expmikr.conj()

        for i in range(nset):
            kiv[i][k1] += weight * lib.dot(vR_dm[i,0:mo_coeff[i][k1][:,mo_occ[i][k1]>0].shape[1]], ao1T.T)
    kiv = mpi.reduce(lib.asarray(kiv))
    if rank==0:
        for i in range(nset):
            for k1 in range(nband):
                kij = lib.einsum('ui,ju->ij', mo_coeff[i][k1][:,mo_occ[i][k1]>0], kiv[i][k1])
                kr = scipy.linalg.solve(kij.conj(), kiv[i][k1])
                vk_kpts[i,k1] = lib.dot(kiv[i][k1].T.conj(),kr)

    if gamma_point(kpts_band) and gamma_point(kpts):
        vk_kpts = vk_kpts.real

    if rank == 0:
        if exxdiv == 'ewald':
            _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts_band=kpts_band)
        return _format_jks(vk_kpts, dm_kpts, input_band, kpts)


def _sync_mydf(mydf):
    mydf.unpack_(comm.bcast(mydf.pack()))
    return mydf

