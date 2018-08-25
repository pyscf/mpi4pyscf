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
    if mol.nbas > 600:
        gen_prescreen = _generate_vj_s8_prescreen
    else:
        gen_prescreen = None
    vj = _eval_jk(mf, mol, dm, hermi, _vj_jobs_s8, gen_prescreen)
    return vj

@mpi.parallel_call
def get_k(mf, mol, dm, hermi=1):
    mf.unpack_(comm.bcast(mf.pack()))
    if mol.nbas > 600:
        gen_prescreen = _generate_vk_s8_prescreen
    else:
        gen_prescreen = None
    vk = _eval_jk(mf, mol, dm, hermi, _vk_jobs_s8, gen_prescreen)
    return vk

def _eval_jk(mf, mol, dm, hermi, gen_jobs, gen_prescreen=None):
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
    vhfopt._dmcondname = None # to skip set_dm in _vhf.direct_bindm

    ngroups = len(bas_groups)
    nbas = mol.nbas
    q_cond = lib.frompointer(vhfopt._this.contents.q_cond, nbas**2)
    q_cond = q_cond.reshape(nbas,nbas)
    dm_cond = mol.condense_to_shell(abs(dm))
    dm_cond = (dm_cond + dm_cond.T) * .5
    if gen_prescreen is None:
        prescreen = lambda *args: True
    else:
        prescreen = gen_prescreen(mf, q_cond, dm_cond, bas_groups)

    # Skip the "set_dm" initialization in function jk.get_jk/direct_bindm.
    # The prescreen function CVHFnrs8_prescreen will search for q_cond and
    # dm_cond over the entire basis. "set_dm" in function jk.get_jk/direct_bindm
    # only creates a subblock of dm_cond which is not compatible with
    # CVHFnrs8_prescreen.
    vhfopt._this.contents.dm_cond = dm_cond.ctypes.data_as(ctypes.c_void_p)

    if 1:
        # To avoid overhead of jk.get_jk, some initialization for
        # _vhf.direct_bindm.
        c_atm = mol._atm.ctypes.data_as(ctypes.c_void_p)
        c_bas = mol._bas.ctypes.data_as(ctypes.c_void_p)
        c_env = mol._env.ctypes.data_as(ctypes.c_void_p)
        c_natm = ctypes.c_int(mol.natm)
        c_nbas = ctypes.c_int(mol.nbas)
        c_ao_loc = ao_loc.ctypes.data_as(ctypes.c_void_p)
        c_comp1 = ctypes.c_int(1)

        intor = mol._add_suffix('int2e')
        cvhfopt = vhfopt._this
        cintopt = vhfopt._cintopt
        cintor = getattr(_vhf.libcvhf, intor)
        fdrv = getattr(_vhf.libcvhf, 'CVHFnr_direct_drv')
        fdot = _vhf._fpointer('CVHFdot_nrs1')
        fjk = (ctypes.c_void_p*8)()
        dmsptr = (ctypes.c_void_p*8)()

        dm_cache = []
        c_dm_cache = []
        for ish0, ish1 in bas_groups:
            for jsh0, jsh1 in bas_groups:
                sub_dm = dm[ao_loc[ish0]:ao_loc[ish1],ao_loc[jsh0]:ao_loc[jsh1]]
                dm_cache.append(numpy.asarray(sub_dm, order='C'))
                c_dm_cache.append(dm_cache[-1].ctypes.data_as(ctypes.c_void_p))
        max_size = max(x.size for x in dm_cache)
        jk_buffer = numpy.empty((8,max_size))
        vjkptr = (ctypes.c_void_p*8)(*[x.ctypes.data_as(ctypes.c_void_p)
                                       for x in jk_buffer])
        nr_dot_lookup = {}
        for key in ((1,0,2,3), (0,1,2,3), (3,2,0,1), (2,3,0,1),
                    (1,0,2,3), (2,3,1,0), (3,2,1,0),
                    (1,2,0,3), (1,3,0,2), (0,2,1,3), (0,3,1,2),
                    (3,0,2,1), (3,1,2,0), (2,0,3,1), (2,1,3,0)):
            f1 = 'CVHFnrs1_%s%s_s1%s%s' % tuple(['ijkl'[x] for x in key])
            nr_dot_lookup[key] = _vhf._fpointer(f1)

    count = skip = 0
    for job_id in mpi.work_stealing_partition(range(njobs)):
        group_ids, recipe = jobs[job_id]
        if not prescreen(*group_ids):
            skip += 1
            continue
        count += 1

        shls_slice = lib.flatten([bas_groups[i] for i in group_ids])
        loc = ao_loc[shls_slice].reshape(4,2)

        if 0: # Avoid the overhead in jk.get_jk
            dms = []
            scripts = []
            for rec in recipe:
                p0, p1 = loc[rec[0]]
                q0, q1 = loc[rec[1]]
                dms.append(numpy.asarray(dm[p0:p1,q0:q1], order='C'))
                scripts.append('ijkl,%s%s->%s%s' % tuple(['ijkl'[x] for x in rec]))
            kparts = jk.get_jk(mol, dms, scripts, shls_slice=shls_slice,
                               vhfopt=vhfopt)
            for i, rec in enumerate(recipe):
                p0, p1 = loc[rec[2]]
                q0, q1 = loc[rec[3]]
                vk[p0:p1,q0:q1] += kparts[i]

        else:
            for i, rec in enumerate(recipe):
                fjk[i] = nr_dot_lookup[rec]
                dmsptr[i] = c_dm_cache[group_ids[rec[0]]*ngroups+group_ids[rec[1]]]
            fdrv(cintor, fdot, fjk, dmsptr, vjkptr,
                 ctypes.c_int(len(recipe)), c_comp1,
                 (ctypes.c_int*8)(*shls_slice), c_ao_loc, cintopt, cvhfopt,
                 c_atm, c_natm, c_bas, c_nbas, c_env)

            for i, rec in enumerate(recipe):
                p0, p1 = loc[rec[2]]
                q0, q1 = loc[rec[3]]
                kpart = numpy.ndarray((p1-p0,q1-q0), buffer=jk_buffer[i])
                vk[p0:p1,q0:q1] += kpart

    # dm_cond's memory will be released when destructing vhfopt. dm_cond is
    # now bound to an nparray. It needs to be detached before deleting
    # vhfopt.
    vhfopt._this.contents.dm_cond = None

    logger.alldebug1(mf, 'job count %d, skip %d', count, skip)
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
    # Avoid huge batch to respect the locality and 8-fold symmetry.
                   nbas_per_atom.mean()*10)
    batch_size = int(min(numpy.prod(batch_size)**.5,
    # Enough workload (60 batches per proc) for MPI processors to utilize the
    # load balance function work_stealing_partition
                         nbas / (mpi.pool.size*60*8)**.25))
    bas_groups = list(lib.prange(0, mol.nbas, batch_size))
    logger.debug1(mol, 'batch_size %d, ngroups = %d',
                  batch_size, len(bas_groups))
    return bas_groups

def _generate_vj_s8_prescreen(mf, q_cond, dm_cond, bas_groups):
    bas_groups = numpy.asarray(bas_groups)
    direct_scf_tol = mf.direct_scf_tol
    def test(q1, dm, q2):
        v = numpy.einsum('ij,ji->', q1, dm)
        return numpy.any(q2*v > .25*direct_scf_tol)
    def vj_screen(i, j, k, l):
        i0, i1, j0, j1, k0, k1, l0, l1 = bas_groups[[i,j,k,l]].ravel()
        #:qijkl = numpy.einsum('ij,kl->ijkl', q_cond[i0:i1,j0:j1], q_cond[k0:k1,l0:l1])
        #:return (numpy.any(numpy.einsum('ijkl,ji->kl', qijkl, dm_cond[j0:j1,i0:i1])
        #:                  > .25*direct_scf_tol) or
        #:        numpy.any(numpy.einsum('ijkl,lk->ij', qijkl, dm_cond[k0:k1,l0:l1])
        #:                  > .25*direct_scf_tol))
        if test(q_cond[i0:i1,j0:j1], dm_cond[j0:j1,i0:i1], q_cond[k0:k1,l0:l1]):
            return True
        elif test(q_cond[k0:k1,l0:l1], dm_cond[l0:l1,k0:k1], q_cond[i0:i1,j0:j1]):
            return True
        else:
            return False
    return vj_screen

def _generate_vk_s8_prescreen(mf, q_cond, dm_cond, bas_groups):
    bas_groups = numpy.asarray(bas_groups)
    direct_scf_tol = mf.direct_scf_tol

    def test(q1, dm, q2):
        v = q1.dot(dm).dot(q2)
        return numpy.any(v > direct_scf_tol)

    def vk_screen(i, j, k, l):
        i0, i1, j0, j1, k0, k1, l0, l1 = bas_groups[[i,j,k,l]].ravel()
        #:qijkl = numpy.einsum('ij,kl->ijkl', q_cond[i0:i1,j0:j1], q_cond[k0:k1,l0:l1])
        #:return (numpy.any(numpy.einsum('ijkl,jk->il', qijkl, dm_cond[j0:j1,i0:i1]) > direct_scf_tol) or
        #:        numpy.any(numpy.einsum('ijkl,jl->ik', qijkl, dm_cond[k0:k1,l0:l1]) > direct_scf_tol) or
        #:        numpy.any(numpy.einsum('ijkl,ik->jl', qijkl, dm_cond[k0:k1,l0:l1]) > direct_scf_tol) or
        #:        numpy.any(numpy.einsum('ijkl,il->jk', qijkl, dm_cond[k0:k1,l0:l1]) > direct_scf_tol))
        if test(q_cond[i0:i1,j0:j1], dm_cond[j0:j1,k0:k1], q_cond[k0:k1,l0:l1]):
            return True
        elif test(q_cond[i0:i1,j0:j1], dm_cond[j0:j1,l0:l1], q_cond[k0:k1,l0:l1].T):
            return True
        elif test(q_cond[i0:i1,j0:j1].T, dm_cond[i0:i1,k0:k1], q_cond[k0:k1,l0:l1]):
            return True
        elif test(q_cond[i0:i1,j0:j1].T, dm_cond[i0:i1,l0:l1], q_cond[k0:k1,l0:l1].T):
            return True
        else:
            return False
    return vk_screen

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

