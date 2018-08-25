#!/usr/bin/env python

import time
import copy
import ctypes
import numpy
from pyscf import lib
from pyscf import scf
from pyscf.scf import jk
from pyscf.scf import _vhf

from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi

comm = mpi.comm
rank = mpi.rank

@mpi.parallel_call
def get_jk(mf, mol, dm, hermi=1):
    '''MPI version of scf.hf.get_jk function'''
    vj = get_j(mf, mol, dm, hermi)
    vk = get_k(mf, mol, dm, hermi)
    return vj, vk

@mpi.parallel_call
def get_j(mf, mol, dm, hermi=1):
    mf.unpack_(comm.bcast(mf.pack()))
    hermi = 1
    nao = dm.shape[0]
    dm = dm + dm.T
    dm[numpy.diag_indices(nao)] *= .5
    dm[numpy.tril_indices(nao, -1)] = 0
    if mf.opt is None:
        mf.opt = mf.init_direct_scf(mol)
    with lib.temporary_env(mf.opt._this.contents,
                           fprescreen=_vhf._fpointer('CVHFnrs8_vj_prescreen')):
        vj = _eval_jk(mf, mol, dm, hermi, _vj_jobs_s8)
    return vj

@mpi.parallel_call
def get_k(mf, mol, dm, hermi=1):
    mf.unpack_(comm.bcast(mf.pack()))
    if mf.opt is None:
        mf.opt = mf.init_direct_scf(mol)
    with lib.temporary_env(mf.opt._this.contents,
                           fprescreen=_vhf._fpointer('CVHFnrs8_vk_prescreen')):
        vk = _eval_jk(mf, mol, dm, hermi, _vk_jobs_s8)
    return vk

def _eval_jk(mf, mol, dm, hermi, gen_jobs):
    ao_loc = mol.ao_loc_nr()
    nao = ao_loc[-1]
    vk = numpy.zeros((nao,nao))

    bas_groups = _partition_bas(mol)
    jobs = gen_jobs(len(bas_groups), hermi)
    njobs = len(jobs)
    logger.debug1(mf, 'njobs %d', njobs)

    if mf.opt is None:
        vhfopt = mf.init_direct_scf(mol)
    else:
        vhfopt = mf.opt
    # Skip the "set_dm" initialization in function jk.get_jk/direct_bindm.
    vhfopt._dmcondname = None
    # Assign the entire dm_cond to vhfopt.  The prescreen function
    # CVHFnrs8_prescreen will search for q_cond and dm_cond over the entire
    # basis. "set_dm" in function jk.get_jk/direct_bindm only creates a
    # subblock of dm_cond which is not compatible with CVHFnrs8_prescreen.
    dm_cond = mol.condense_to_shell(abs(dm))
    dm_cond = (dm_cond + dm_cond.T) * .5
    vhfopt._this.contents.dm_cond = dm_cond.ctypes.data_as(ctypes.c_void_p)

    for job_id in mpi.work_stealing_partition(range(njobs)):
        group_ids, recipe = jobs[job_id]

        shls_slice = lib.flatten([bas_groups[i] for i in group_ids])
        loc = ao_loc[shls_slice].reshape(4,2)

        dms = []
        for rec in recipe:
            p0, p1 = loc[rec[0]]
            q0, q1 = loc[rec[1]]
            dms.append(numpy.asarray(dm[p0:p1,q0:q1], order='C'))
        scripts = ['ijkl,%s%s->%s%s' % tuple(['ijkl'[x] for x in rec])
                   for rec in recipe]
        kparts = jk.get_jk(mol, dms, scripts, shls_slice=shls_slice,
                           vhfopt=vhfopt)
        for i, rec in enumerate(recipe):
            p0, p1 = loc[rec[2]]
            q0, q1 = loc[rec[3]]
            vk[p0:p1,q0:q1] += kparts[i]

    # dm_cond's memory will be released when destructing vhfopt. dm_cond is
    # now bound to an nparray. It needs to be detached before deleting
    # vhfopt.
    vhfopt._this.contents.dm_cond = None

    vk = mpi.reduce(vk)
    if rank == 0 and hermi:
        vk = lib.hermi_triu(vk, hermi, inplace=True)
    return vk

def _partition_bas(mol):
    aoslice = mol.aoslice_by_atom()
    nbas_per_atom = aoslice[:,1] - aoslice[:,0]
    nthreads = lib.num_threads()
    nbas = mol.nbas
    # Enough workload in each batch (here 600 pairs of shells per thread) for
    # OMP load balance.
    batch_size = ((nthreads * 600)**.5,
    # Avoid huge batch, to respect the locality and 8-fold symmetry.
                   nbas_per_atom.mean()*10)
    batch_size = int(min(numpy.prod(batch_size)**.5,
    # Enough workload (60 batches per proc) for MPI processors to utilize the
    # load balance function work_stealing_partition
                         nbas / (mpi.pool.size*60*8)**.25))
    bas_groups = list(lib.prange(0, mol.nbas, batch_size))
    logger.debug1(mol, 'batch_size %d, ngroups = %d',
                  batch_size, len(bas_groups))
    return bas_groups

def _vj_jobs_s8(ngroups, hermi=1):
    jobs = []
    if hermi:
        recipe = ((1,0,2,3), (3,2,0,1))
    else:
        recipe = ((1,0,2,3), (0,1,2,3), (3,2,0,1), (2,3,0,1))
    for ip in range(ngroups):
        for jp in range(ip):
            for kp in range(ip):
                for lp in range(kp):
                    jobs.append(((ip, jp, kp, lp), recipe))

    # ip > jp, ip > kp, kp == lp
    if hermi:
        recipe = ((1,0,2,3), (2,3,0,1))
    else:
        recipe = ((1,0,2,3), (0,1,2,3), (2,3,0,1))
    for ip in range(ngroups):
        for jp in range(ip):
            for kp in range(ip):
                lp = kp
                jobs.append(((ip, jp, kp, lp), recipe))

    # ip == kp and ip > jp and kp > lp
    if hermi:
        recipe = ((1,0,2,3),)
    else:
        recipe = ((1,0,2,3), (0,1,2,3))
    for ip in range(ngroups):
        for jp in range(ip):
            kp = ip
            for lp in range(kp):
                jobs.append(((ip, jp, kp, lp), recipe))

    # ip == jp and ip >= kp
    if hermi:
        recipe = ((1,0,2,3), (3,2,1,0))
    else:
        recipe = ((1,0,2,3), (2,3,1,0), (3,2,1,0))
    for ip in range(ngroups):
        jp = ip
        for kp in range(ip+1):
            for lp in range(kp):
                jobs.append(((ip, jp, kp, lp), recipe))

    # ip == jp and ip > kp and kp == lp
    recipe = ((1,0,2,3), (3,2,0,1))
    for ip in range(ngroups):
        jp = ip
        for kp in range(ip):
            lp = kp
            jobs.append(((ip, jp, kp, lp), recipe))

    # ip == jp == kp == lp
    recipe = ((1,0,2,3),)
    for ip in range(ngroups):
        jobs.append(((ip, ip, ip, ip), recipe))
    return jobs

def _vk_jobs_s8(ngroups, hermi=1):
    jobs = []
    if hermi:
        recipe = ((1,2,0,3), (1,3,0,2), (0,2,1,3), (0,3,1,2),
                  (3,0,2,1),            (2,0,3,1)           )
    else:
        recipe = ((1,2,0,3), (1,3,0,2), (0,2,1,3), (0,3,1,2),
                  (3,0,2,1), (3,1,2,0), (2,0,3,1), (2,1,3,0))
    for ip in range(ngroups):
        for jp in range(ip):
            for kp in range(ip):
                for lp in range(kp):
                    jobs.append(((ip, jp, kp, lp), recipe))

    # ip > jp, ip > kp, kp == lp
    if hermi:
        recipe = ((1,2,0,3), (0,2,1,3),            (2,0,3,1))
    else:
        recipe = ((1,2,0,3), (0,2,1,3), (2,1,3,0), (2,0,3,1))
    for ip in range(ngroups):
        for jp in range(ip):
            for kp in range(ip):
                lp = kp
                jobs.append(((ip, jp, kp, lp), recipe))

    # ip == kp and ip > jp and kp > lp
    recipe = ((1,2,0,3), (0,2,1,3), (1,3,0,2), (0,3,1,2))
    for ip in range(ngroups):
        for jp in range(ip):
            kp = ip
            for lp in range(kp):
                jobs.append(((ip, jp, kp, lp), recipe))

    # ip == jp and ip >= kp
    if hermi:
        recipe = ((1,2,0,3), (1,3,0,2),            (3,1,2,0))
    else:
        recipe = ((1,2,0,3), (1,3,0,2), (2,1,3,0), (3,1,2,0))
    for ip in range(ngroups):
        jp = ip
        for kp in range(ip+1):
            for lp in range(kp):
                jobs.append(((ip, jp, kp, lp), recipe))

    # ip == jp and ip > kp and kp == lp
    if hermi:
        recipe = ((1,2,0,3),)
    else:
        recipe = ((1,2,0,3), (3,0,2,1))
    for ip in range(ngroups):
        jp = ip
        for kp in range(ip):
            lp = kp
            jobs.append(((ip, jp, kp, lp), recipe))

    # ip == jp == kp == lp
    recipe = ((1,2,0,3),)
    for ip in range(ngroups):
        jobs.append(((ip, ip, ip, ip), recipe))
    return jobs


@mpi.register_class
class RHF(scf.hf.RHF):

    get_jk = get_jk
    get_j = get_j
    get_k = get_k

    def pack(self):
        return {'verbose'   : self.verbose,
                'direct_scf_tol': self.direct_scf_tol}
    def unpack_(self, hf_dic):
        self.__dict__.update(hf_dic)
        return self

    def dump_flags(self):
        if rank == 0:
            scf.hf.RHF.dump_flags(self)
        return self
    def sanity_check(self):
        if rank == 0:
            scf.hf.RHF.sanity_check(self)
        return self

