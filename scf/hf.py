#!/usr/bin/env python

import time
import copy
import ctypes
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import ao2mo
from pyscf.scf import hf
from pyscf.scf import jk
from pyscf.scf import _vhf

from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi

comm = mpi.comm
rank = mpi.rank

@mpi.parallel_call
def get_jk(mol_or_mf, dm, hermi=1):
    '''MPI version of scf.hf.get_jk function'''
    vj = get_j(mol_or_mf, dm, hermi)
    vk = get_k(mol_or_mf, dm, hermi)
    return vj, vk

@mpi.parallel_call
def get_j(mol_or_mf, dm, hermi=1):
    if isinstance(mol_or_mf, gto.mole.Mole):
        mf = hf.SCF(mol_or_mf).view(SCF)
    else:
        mf = mol_or_mf

    hermi = 1
    nao = dm.shape[0]
    dm = dm + dm.T
    dm[numpy.diag_indices(nao)] *= .5
    dm[numpy.tril_indices(nao, -1)] = 0

    mf.unpack_(comm.bcast(mf.pack()))
    if mf.opt is None:
        mf.opt = mf.init_direct_scf()
    with lib.temporary_env(mf.opt._this.contents,
                           fprescreen=_vhf._fpointer('CVHFnrs8_vj_prescreen')):
        vj = _eval_jk(mf, dm, hermi, _vj_jobs_s8)
    return vj

@mpi.parallel_call
def get_k(mol_or_mf, dm, hermi=1):
    if isinstance(mol_or_mf, gto.mole.Mole):
        mf = hf.SCF(mol_or_mf).view(SCF)
    else:
        mf = mol_or_mf

    mf.unpack_(comm.bcast(mf.pack()))
    if mf.opt is None:
        mf.opt = mf.init_direct_scf()
    with lib.temporary_env(mf.opt._this.contents,
                           fprescreen=_vhf._fpointer('CVHFnrs8_vk_prescreen')):
        vk = _eval_jk(mf, dm, hermi, _vk_jobs_s8)
    return vk

def _eval_jk(mf, dm, hermi, gen_jobs):
    mol = mf.mol
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
    # Assign the entire dm_cond to vhfopt.
    # The prescreen function CVHFnrs8_prescreen will index q_cond and dm_cond
    # over the entire basis.  "set_dm" in function jk.get_jk/direct_bindm only
    # creates a subblock of dm_cond which is not compatible with
    # CVHFnrs8_prescreen.
    vhfopt.set_dm(dm, mol._atm, mol._bas, mol._env)
    # Then skip the "set_dm" initialization in function jk.get_jk/direct_bindm.
    vhfopt._dmcondname = None

    for job_id in mpi.work_stealing_partition(range(njobs)):
        group_ids, recipe = jobs[job_id]

        shls_slice = lib.flatten([bas_groups[i] for i in group_ids])
        loc = ao_loc[shls_slice].reshape(4,2)

        dms = []
        for rec in recipe:
            p0, p1 = loc[rec[0]]
            q0, q1 = loc[rec[1]]
            dms.append(dm[p0:p1,q0:q1])
        scripts = ['ijkl,%s%s->%s%s' % tuple(['ijkl'[x] for x in rec])
                   for rec in recipe]
        kparts = jk.get_jk(mol, dms, scripts, shls_slice=shls_slice,
                           vhfopt=vhfopt)
        for i, rec in enumerate(recipe):
            p0, p1 = loc[rec[2]]
            q0, q1 = loc[rec[3]]
            vk[p0:p1,q0:q1] += kparts[i]

    vk = mpi.reduce(vk)
    if rank == 0 and hermi:
        vk = lib.hermi_triu(vk, hermi, inplace=True)
    return vk

def _partition_bas(mol):
    ao_loc = mol.ao_loc_nr()
    nao = ao_loc[-1]
    ngroups = max((mpi.pool.size*60*8)**.25, 9)
    blksize = max(60, min(nao / ngroups, 600))
    groups = ao2mo.outcore.balance_partition(ao_loc, blksize)
    bas_groups = [x[:2] for x in groups]
    logger.debug1(mol, 'mpi.size %d, blksize = %d, ngroups = %d',
                  mpi.pool.size, blksize, len(bas_groups))
    logger.debug2(mol, 'bas_groups = %s', bas_groups)
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
class SCF(hf.SCF):

    @lib.with_doc(hf.SCF.get_jk.__doc__)
    def get_jk(self, mol, dm, hermi=1):
        assert(mol is self.mol)
        return get_jk(self, dm, hermi)

    @lib.with_doc(hf.SCF.get_j.__doc__)
    def get_j(self, mol, dm, hermi=1):
        assert(mol is self.mol)
        return get_j(self, dm, hermi)

    @lib.with_doc(hf.SCF.get_k.__doc__)
    def get_k(self, mol, dm, hermi=1):
        assert(mol is self.mol)
        return get_k(self, dm, hermi)

    def pack(self):
        return {'verbose': self.verbose,
                'direct_scf_tol': self.direct_scf_tol}
    def unpack_(self, mf_dic):
        self.__dict__.update(mf_dic)
        return self

    def dump_flags(self):
        if rank == 0:
            hf.SCF.dump_flags(self)
        return self
    def sanity_check(self):
        if rank == 0:
            hf.SCF.sanity_check(self)
        return self

class RHF(SCF):
    pass
