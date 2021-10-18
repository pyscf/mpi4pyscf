#!/usr/bin/env python

import time
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

from pyscf import __config__
BLKSIZE_MIN = getattr(__config__, 'scf_hf_BLKSIZE_MIN', 60)
BLKSIZE_MAX = getattr(__config__, 'scf_hf_BLKSIZE_MAX', 800)


@lib.with_doc(hf.get_jk.__doc__)
@mpi.parallel_call(skip_args=[1])
def get_jk(mol_or_mf=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
    '''MPI version of scf.hf.get_jk function'''
    #vj = get_j(mol_or_mf, dm, hermi)
    #vk = get_k(mol_or_mf, dm, hermi)
    if isinstance(mol_or_mf, gto.mole.Mole):
        mf = hf.SCF(mol_or_mf).view(SCF)
    else:
        mf = mol_or_mf

    # dm may be too big for mpi4py library to serialize. Broadcast dm here.
    if any(comm.allgather(dm is mpi.Message.SkippedArg)):
        dm = mpi.bcast_tagged_array(dm)

    mf.unpack_(comm.bcast(mf.pack()))
    if mf.opt is None:
        mf.opt = mf.init_direct_scf()

    if omega is None:
        vj, vk = _eval_jk(mf, dm, hermi, _jk_jobs_s8)
    else:
        with mf.mol.with_range_coulomb(omega):
            vj, vk = _eval_jk(mf, dm, hermi, _jk_jobs_s8)

    if rank == 0:
        for i in range(vj.shape[0]):
            lib.hermi_triu(vj[i], 1, inplace=True)
    return vj.reshape(dm.shape), vk.reshape(dm.shape)

@lib.with_doc(hf.SCF.get_j.__doc__)
@mpi.parallel_call(skip_args=[1])
def get_j(mol_or_mf=None, dm=None, hermi=1, omega=None):
    if isinstance(mol_or_mf, gto.mole.Mole):
        mf = hf.SCF(mol_or_mf).view(SCF)
    else:
        mf = mol_or_mf

    # dm may be too big for mpi4py library to serialize. Broadcast dm here.
    if any(comm.allgather(dm is mpi.Message.SkippedArg)):
        dm = mpi.bcast_tagged_array(dm)

    mf.unpack_(comm.bcast(mf.pack()))
    if mf.opt is None:
        mf.opt = mf.init_direct_scf()
    with lib.temporary_env(mf.opt._this.contents,
                           fprescreen=_vhf._fpointer('CVHFnrs8_vj_prescreen')):
        hermi = 1
        if omega is None:
            vj = _eval_jk(mf, dm, hermi, _vj_jobs_s8)
        else:
            with mf.mol.with_range_coulomb(omega):
                vj = _eval_jk(mf, dm, hermi, _vj_jobs_s8)
    return vj.reshape(dm.shape)

@lib.with_doc(hf.SCF.get_k.__doc__)
@mpi.parallel_call(skip_args=[1])
def get_k(mol_or_mf=None, dm=None, hermi=1, omega=None):
    if isinstance(mol_or_mf, gto.mole.Mole):
        mf = hf.SCF(mol_or_mf).view(SCF)
    else:
        mf = mol_or_mf

    # dm may be too big for mpi4py library to serialize. Broadcast dm here.
    if any(comm.allgather(dm is mpi.Message.SkippedArg)):
        dm = mpi.bcast_tagged_array(dm)

    mf.unpack_(comm.bcast(mf.pack()))
    if mf.opt is None:
        mf.opt = mf.init_direct_scf()
    with lib.temporary_env(mf.opt._this.contents,
                           fprescreen=_vhf._fpointer('CVHFnrs8_vk_prescreen')):
        if omega is None:
            vk = _eval_jk(mf, dm, hermi, _vk_jobs_s8)
        else:
            with mf.mol.with_range_coulomb(omega):
                vk = _eval_jk(mf, dm, hermi, _vk_jobs_s8)
    return vk.reshape(dm.shape)

def _eval_jk(mf, dm, hermi, gen_jobs):
    cpu0 = (logger.process_clock(), logger.perf_counter())
    mol = mf.mol
    ao_loc = mol.ao_loc_nr()
    nao = ao_loc[-1]

    bas_groups = _partition_bas(mol)
    jobs = gen_jobs(len(bas_groups), hermi)
    njobs = len(jobs)
    logger.debug1(mf, 'njobs %d', njobs)

    # Each job has multiple recipes.
    n_recipes = len(jobs[0][1:])
    dm = numpy.asarray(dm).reshape(-1,nao,nao)
    n_dm = dm.shape[0]
    vk = numpy.zeros((n_recipes,n_dm,nao,nao))

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

    logger.timer_debug1(mf, 'get_jk initialization', *cpu0)
    for job_id in mpi.work_stealing_partition(range(njobs)):
        group_ids = jobs[job_id][0]
        recipes = jobs[job_id][1:]

        shls_slice = lib.flatten([bas_groups[i] for i in group_ids])
        loc = ao_loc[shls_slice].reshape(4,2)

        dm_blks = []
        for i_dm in range(n_dm):
            for ir, recipe in enumerate(recipes):
                for i, rec in enumerate(recipe):
                    p0, p1 = loc[rec[0]]
                    q0, q1 = loc[rec[1]]
                    dm_blks.append(dm[i_dm,p0:p1,q0:q1])
        scripts = ['ijkl,%s%s->%s%s' % tuple(['ijkl'[x] for x in rec])
                   for recipe in recipes
                       for rec in recipe] * n_dm

        kparts = jk.get_jk(mol, dm_blks, scripts, shls_slice=shls_slice,
                           vhfopt=vhfopt)

        for i_dm in range(n_dm):
            for ir, recipe in enumerate(recipes):
                for i, rec in enumerate(recipe):
                    p0, p1 = loc[rec[2]]
                    q0, q1 = loc[rec[3]]
                    vk[ir,i_dm,p0:p1,q0:q1] += kparts[i]
                # Pop the results of one recipe
                kparts = kparts[i+1:]

    vk = mpi.reduce(vk)
    if rank == 0:
        if hermi:
            for i in range(n_recipes):
                for j in range(n_dm):
                    lib.hermi_triu(vk[i,j], hermi, inplace=True)
    else:
        # Zero out vk on workers. If reduce(get_jk()) is called twice,
        # non-zero vk on workers can cause error.
        vk[:] = 0
    logger.timer(mf, 'get_jk', *cpu0)
    return vk

def _partition_bas(mol):
    ao_loc = mol.ao_loc_nr()
    nao = ao_loc[-1]
    ngroups = max((mpi.pool.size*50*8)**.25, 9)
    blksize = max(BLKSIZE_MIN, min(nao / ngroups, BLKSIZE_MAX))
    groups = ao2mo.outcore.balance_partition(ao_loc, blksize)
    bas_groups = [x[:2] for x in groups]
    logger.debug1(mol, 'mpi.size %d, blksize = %d, ngroups = %d',
                  mpi.pool.size, blksize, len(bas_groups))
    logger.debug2(mol, 'bas_groups = %s', bas_groups)
    return bas_groups

def _vj_jobs_s8(ngroups, hermi=1):
    jobs = []
    recipe = ((1,0,2,3), (0,1,2,3), (3,2,0,1), (2,3,0,1))
    for ip in range(ngroups):
        for jp in range(ip):
            for kp in range(ip):
                for lp in range(kp):
                    jobs.append(((ip, jp, kp, lp), recipe))

    # ip > jp, ip > kp, kp == lp
    recipe = ((1,0,2,3), (0,1,2,3), (2,3,0,1))
    for ip in range(ngroups):
        for jp in range(ip):
            for kp in range(ip):
                lp = kp
                jobs.append(((ip, jp, kp, lp), recipe))

    # ip > jp and ip == kp and kp > lp
    recipe = ((1,0,2,3), (0,1,2,3))
    for ip in range(ngroups):
        for jp in range(ip):
            kp = ip
            for lp in range(kp):
                jobs.append(((ip, jp, kp, lp), recipe))

    # ip == jp and ip >= kp and kp > lp
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

    # ip > jp and ip == kp and kp > lp
    recipe = ((1,2,0,3), (0,2,1,3), (1,3,0,2), (0,3,1,2))
    for ip in range(ngroups):
        for jp in range(ip):
            kp = ip
            for lp in range(kp):
                jobs.append(((ip, jp, kp, lp), recipe))

    # ip == jp and ip >= kp and kp > lp
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

def _jk_jobs_s8(ngroups, hermi=1):
    j_jobs = _vj_jobs_s8(ngroups)
    k_jobs = _vk_jobs_s8(ngroups, hermi)
    njobs = len(j_jobs)
    assert(len(k_jobs) == len(j_jobs))
    jk_jobs = [(group, j_recipe, k_jobs[i][1])
               for i, (group, j_recipe) in enumerate(j_jobs)]
    return jk_jobs


@mpi.register_class
class SCF(hf.SCF):

    @lib.with_doc(hf.SCF.get_jk.__doc__)
    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True, omega=None):
        assert mol is None or mol is self.mol
        return get_jk(self, dm, hermi, omega=omega)

    @lib.with_doc(hf.SCF.get_j.__doc__)
    def get_j(self, mol=None, dm=None, hermi=1, omega=None):
        assert mol is None or mol is self.mol
        return get_j(self, dm, hermi, omega)

    @lib.with_doc(hf.SCF.get_k.__doc__)
    def get_k(self, mol=None, dm=None, hermi=1, omega=None):
        assert mol is None or mol is self.mol
        return get_k(self, dm, hermi, omega)

    def pack(self):
        return {'verbose': self.verbose,
                'direct_scf_tol': self.direct_scf_tol}
    def unpack_(self, mf_dic):
        self.__dict__.update(mf_dic)
        return self

    def dump_flags(self, verbose=None):
        mpi_info = mpi.platform_info()
        if rank == 0:
            hf.SCF.dump_flags(self, verbose)
            lib.logger.debug(self, 'MPI info (rank, host, pid)  %s', mpi_info)
        return self

class RHF(SCF):
    pass
