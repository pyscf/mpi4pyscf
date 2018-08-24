#!/usr/bin/env python

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
    vj = _eval_jk(mf, mol, dm, hermi, _vj_jobs_s8, _generate_vj_s8_prescreen)
    return vj

@mpi.parallel_call
def get_k(mf, mol, dm, hermi=1):
    mf.unpack_(comm.bcast(mf.pack()))
    vk = _eval_jk(mf, mol, dm, hermi, _vk_jobs_s8, _generate_vk_s8_prescreen)
    return vk

def _eval_jk(mf, mol, dm, hermi, gen_jobs, gen_prescreen):
    ao_loc = mol.ao_loc_nr()
    nao = ao_loc[-1]
    vk = numpy.zeros((nao,nao))

    bas_groups = _partition_bas(mol)
    jobs = gen_jobs(len(bas_groups), hermi)
    njobs = len(jobs)

    vhfopt = mf.opt
    if vhfopt is None:
        intor = mol._add_suffix('int2e')
        vhfopt = _vhf.VHFOpt(mol, 'int2e', 'CVHFnrs8_prescreen',
                             'CVHFsetnr_direct_scf')
    else:
        vhfopt = copy.copy(vhfopt)
        vhfopt._dmcondname = None # to skip set_dm in _vhf.direct_bindm

    # _vhf.direct_bindm does not support prescreen in pyscf-1.6
    vhfopt._this.contents.fprescreen = _vhf._fpointer('CVHFnoscreen')

    ngroups = len(bas_groups)
    nbas = mol.nbas
    q_cond0 = lib.frompointer(vhfopt._this.contents.q_cond, nbas**2)
    q_cond0 = q_cond0.reshape(nbas,nbas)
    q_cond = numpy.empty((ngroups,ngroups))
    dm_cond = numpy.empty((ngroups,ngroups))
    for ip, (ish0, ish1) in enumerate(bas_groups):
        for jp, (jsh0, jsh1) in enumerate(bas_groups):
            i0 = ao_loc[ish0]
            i1 = ao_loc[ish1]
            j0 = ao_loc[jsh0]
            j1 = ao_loc[jsh1]
            q_cond[ip,jp] = abs(q_cond0[ish0:ish1,jsh0:jsh1]).max()
            dm_cond[ip,jp] = abs(dm[i0:i1,j0:j1]).max()

    prescreen = gen_prescreen(q_cond, dm_cond, mf.direct_scf_tol)

    IJKL = 'ijkl'
    for job_id in mpi.work_stealing_partition(range(njobs)):
        job = jobs[job_id]
        group_ids = job[0]
        if not prescreen(*group_ids):
            continue

        shls_slice = lib.flatten([bas_groups[i] for i in group_ids])
        loc = ao_loc[shls_slice].reshape(4,2)
        dms = []
        scripts = []
        for script in job[1:]:
            p0, p1 = loc[script[0]]
            q0, q1 = loc[script[1]]
            dms.append(dm[p0:p1,q0:q1])
            scripts.append('ijkl,%s%s->%s%s' % tuple([IJKL[x] for x in script]))

        kparts = jk.get_jk(mol, dms, scripts, shls_slice=shls_slice,
                           vhfopt=vhfopt)

        for i, script in enumerate(job[1:]):
            p0, p1 = loc[script[2]]
            q0, q1 = loc[script[3]]
            vk[p0:p1,q0:q1] += kparts[i]

    vk = mpi.reduce(vk)
    if rank == 0 and hermi:
        vk = lib.hermi_triu(vk, hermi, inplace=True)
    return vk

def _partition_bas(mol):
    aoslice = mol.aoslice_by_atom()
    nbas_per_atom = aoslice[1:,1] - aoslice[:-1,0]
    nthreads = lib.num_threads()
    nbas = mol.nbas
    # For OMP load balance, each thread at least evaluates 30 pairs of shells.
    batch_size = min(int((nthreads * 30)**.5),
                     nbas_per_atom.max() * 2)
    bas_groups = list(lib.prange(0, mol.nbas, batch_size))
    return bas_groups

def _generate_vj_s8_prescreen(q_cond, dm_cond, direct_scf_cutoff):
    def vj_screen(i, j, k, l):
        qijkl = q_cond[i,j] * q_cond[k,l]
        dmin = direct_scf_cutoff / qijkl
        return (qijkl > direct_scf_cutoff and
                (4*dm_cond[j,i] > dmin) or (4*dm_cond[l,k] > dmin))
    return vj_screen

def _generate_vk_s8_prescreen(q_cond, dm_cond, direct_scf_cutoff):
    def vk_screen(i, j, k, l):
        qijkl = q_cond[i,j] * q_cond[k,l]
        dmin = direct_scf_cutoff / qijkl
        return (qijkl > direct_scf_cutoff and
                (dm_cond[j,k] > dmin) or
                (dm_cond[j,l] > dmin) or
                (dm_cond[i,k] > dmin) or
                (dm_cond[i,l] > dmin))
    return vk_screen

def _vj_jobs_s8(ngroups, hermi=1):
    jobs = []
    for ip in range(ngroups):
        for jp in range(ip):
            for kp in range(ip):
                for lp in range(kp):
                    jobs.append(((ip, jp, kp, lp),
                                 #(1,0,2,3), (0,1,2,3), (3,2,0,1), (2,3,0,1)))
                                 (1,0,2,3), (3,2,0,1)))

                # ip > jp, ip > kp, kp == lp
                lp = kp
                jobs.append(((ip, jp, kp, lp),
                             #(1,0,2,3), (0,1,2,3),            (2,3,0,1)))
                             (1,0,2,3), (2,3,0,1)))

            # ip == kp and ip > jp and kp > lp
            kp = ip
            for lp in range(kp):
                jobs.append(((ip, jp, kp, lp),
                             #(1,0,2,3), (0,1,2,3)))
                             (1,0,2,3)))

        # ip == jp and ip >= kp
        jp = ip
        for kp in range(ip+1):
            for lp in range(kp):
                jobs.append(((ip, jp, kp, lp),
                             #(1,0,2,3),            (2,3,1,0), (3,2,1,0)))
                             (1,0,2,3), (3,2,1,0)))

        # ip == jp and ip > kp and kp == lp
        jp = ip
        for kp in range(ip):
            lp = kp
            jobs.append(((ip, jp, kp, lp),
                         (1,0,2,3), (3,2,0,1)))

        # ip == jp == kp == lp
        jobs.append(((ip, ip, ip, ip),
                     (1,0,2,3)))
    return jobs

def _vk_jobs_s8(ngroups, hermi=1):
    jobs = []
    for ip in range(ngroups):
        for jp in range(ip):
            for kp in range(ip):
                for lp in range(kp):
                    if hermi:
                        jobs.append(((ip, jp, kp, lp),
                                     (1,2,0,3), (1,3,0,2), (0,2,1,3), (0,3,1,2),
                                     (3,0,2,1),            (2,0,3,1)           ))
                    else:
                        jobs.append(((ip, jp, kp, lp),
                                     (1,2,0,3), (1,3,0,2), (0,2,1,3), (0,3,1,2),
                                     (3,0,2,1), (3,1,2,0), (2,0,3,1), (2,1,3,0)))

                # ip > jp, ip > kp, kp == lp
                lp = kp
                if hermi:
                    jobs.append(((ip, jp, kp, lp),
                                 (1,2,0,3), (0,2,1,3),            (2,0,3,1)))
                else:
                    jobs.append(((ip, jp, kp, lp),
                                 (1,2,0,3), (0,2,1,3), (2,1,3,0), (2,0,3,1)))

            # ip == kp and ip > jp and kp > lp
            kp = ip
            for lp in range(kp):
                jobs.append(((ip, jp, kp, lp),
                             (1,2,0,3), (0,2,1,3), (1,3,0,2), (0,3,1,2)))

        # ip == jp and ip >= kp
        jp = ip
        for kp in range(ip+1):
            for lp in range(kp):
                if hermi:
                    jobs.append(((ip, jp, kp, lp),
                                 (1,2,0,3), (1,3,0,2),            (3,1,2,0)))
                else:
                    jobs.append(((ip, jp, kp, lp),
                                 (1,2,0,3), (1,3,0,2), (2,1,3,0), (3,1,2,0)))

        # ip == jp and ip > kp and kp == lp
        jp = ip
        for kp in range(ip):
            lp = kp
            if hermi:
                jobs.append(((ip, jp, kp, lp),
                             (1,2,0,3)))
            else:
                jobs.append(((ip, jp, kp, lp),
                             (1,2,0,3), (3,0,2,1)))

        # ip == jp == kp == lp
        jobs.append(((ip, ip, ip, ip),
                     (1,2,0,3)))
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

