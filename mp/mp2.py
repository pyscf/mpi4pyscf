#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


'''
RMP2
'''

import time
from functools import reduce
import copy
import numpy
from pyscf import gto
from pyscf import lib
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf.mp import mp2
from pyscf import __config__

from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi

comm = mpi.comm
rank = mpi.rank

BLKMIN = 4


@mpi.parallel_call
def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=False,
           verbose=logger.NOTE):
    _sync_(mp)
    if mo_energy is None or mo_coeff is None:
        if mp.mo_energy is None or mp.mo_coeff is None:
            raise RuntimeError('mo_coeff, mo_energy are not initialized.\n'
                               'You may need to call mf.kernel() to generate them.')
        mo_coeff = None
        mo_energy = mp2._mo_energy_without_core(mp, mp.mo_energy)
    else:
        assert(mp.frozen is 0 or mp.frozen is None)

    eris = getattr(mp, '_eris', None)
    if eris is None:
        mp.ao2mo(mo_coeff)
        eris = mp._eris

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    oloc0, oloc1 = _task_location(nocc, rank)
    nocc_seg = oloc1 - oloc0
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]

    if with_t2:
        t2 = numpy.empty((nocc_seg,nocc,nvir,nvir), dtype=eris.ovov.dtype)
    else:
        t2 = None

    emp2 = 0
    for i in range(nocc):
        gi = numpy.asarray(eris.ovov[i])
        gi = gi.reshape(nvir,nocc_seg,nvir).transpose(1,2,0)
        t2i = gi.conj() / (eia[oloc0:oloc1,:,None] + eia[i])
        emp2 += numpy.einsum('jab,jab', t2i, gi) * 2
        emp2 -= numpy.einsum('jab,jba', t2i, gi)
        if with_t2:
            t2[:,i] = t2i

    emp2 = mpi.comm.allreduce(emp2)
    return emp2.real, t2


@mpi.register_class_without__init__
class MP2(mp2.MP2):

    def pack(self):
        return {'verbose'   : self.verbose,
                'max_memory': self.max_memory,
                'frozen'    : self.frozen,
                'mo_energy' : self.mo_energy,
                'mo_coeff'  : self.mo_coeff,
                'mo_occ'    : self.mo_occ,
                '_nocc'     : self._nocc,
                '_nmo'      : self._nmo}
    def unpack_(self, mp2_dic):
        self.__dict__.update(mp2_dic)
        return self

    def dump_flags(self):
        if rank == 0:
            mp2.MP2.dump_flags(self)
        return self
    def sanity_check(self):
        if rank == 0:
            mp2.MP2.sanity_check(self)
        return self

    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=False):
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        self.e_corr, self.t2 = kernel(self, mo_energy, mo_coeff,
                                      eris, with_t2, self.verbose)
        if rank == 0:
            self._finalize()
        return self.e_corr, self.t2

    def ao2mo(self, mo_coeff=None):
        return _make_eris(self, mo_coeff, verbose=self.verbose)

RMP2 = MP2


@mpi.parallel_call
def _make_eris(mp, mo_coeff=None, verbose=None):
    log = logger.new_logger(mp, verbose)
    time0 = (time.clock(), time.time())

    log.debug('transform (ia|jb) outcore')
    mol = mp.mol
    nocc = mp.nocc
    nmo = mp.nmo
    nvir = nmo - nocc

    eris = mp2._ChemistsERIs(mp, mo_coeff)
    nao = eris.mo_coeff.shape[0]
    assert(nvir <= nao)
    orbo = eris.mo_coeff[:,:nocc]
    orbv = numpy.asarray(eris.mo_coeff[:,nocc:], order='F')
    eris.feri = lib.H5TmpFile()

    int2e = mol._add_suffix('int2e')
    ao2mopt = _ao2mo.AO2MOpt(mol, int2e, 'CVHFnr_schwarz_cond',
                             'CVHFsetnr_direct_scf')
    fint = gto.moleintor.getints4c

    ntasks = mpi.pool.size
    olocs = [_task_location(nocc, task_id) for task_id in range(ntasks)]
    oloc0, oloc1 = olocs[rank]
    nocc_seg = oloc1 - oloc0
    log.debug2('olocs %s', olocs)

    ao_loc = mol.ao_loc_nr()
    task_sh_locs = lib.misc._balanced_partition(ao_loc, ntasks)
    log.debug2('task_sh_locs %s', task_sh_locs)
    ao_sh0 = task_sh_locs[rank]
    ao_sh1 = task_sh_locs[rank+1]
    ao_loc0 = ao_loc[ao_sh0]
    ao_loc1 = ao_loc[ao_sh1]
    nao_seg = ao_loc1 - ao_loc0
    orbo_seg = orbo[ao_loc0:ao_loc1]

    mem_now = lib.current_memory()[0]
    max_memory = max(0, mp.max_memory - mem_now)
    dmax = numpy.sqrt(max_memory*.9e6/8/((nao+nocc)*(nao_seg+nocc)))
    dmax = min(nao//4+2, max(BLKMIN, min(comm.allgather(dmax))))
    sh_ranges = ao2mo.outcore.balance_partition(ao_loc, dmax)
    sh_ranges = comm.bcast(sh_ranges)
    dmax = max(x[2] for x in sh_ranges)
    eribuf = numpy.empty((nao,dmax,dmax,nao_seg))
    ftmp = lib.H5TmpFile()
    log.debug('max_memory %s MB (dmax = %s) required disk space %g MB',
              max_memory, dmax, nocc*nocc_seg*(nao*(nao+dmax)/2+nvir**2)*8/1e6)

    def save(count, tmp_xo):
        di, dj = tmp_xo.shape[2:4]
        tmp_xo = [tmp_xo[p0:p1] for p0, p1 in olocs]
        tmp_xo = mpi.alltoall(tmp_xo, split_recvbuf=True)
        tmp_xo = sum(tmp_xo).reshape(nocc_seg,nocc,di,dj)
        ftmp[str(count)+'b'] = tmp_xo

        tmp_ox = mpi.alltoall([tmp_xo[:,p0:p1] for p0, p1 in olocs],
                              split_recvbuf=True)
        tmp_ox = [tmp_ox[i].reshape(p1-p0,nocc_seg,di,dj)
                  for i, (p0,p1) in enumerate(olocs)]
        ftmp[str(count)+'a'] = numpy.vstack(tmp_ox)

    jk_blk_slices = []
    count = 0
    time1 = time0
    with lib.call_in_background(save) as bg_save:
        for ip, (ish0, ish1, ni) in enumerate(sh_ranges):
            for jsh0, jsh1, nj in sh_ranges[:ip+1]:
                i0, i1 = ao_loc[ish0], ao_loc[ish1]
                j0, j1 = ao_loc[jsh0], ao_loc[jsh1]
                jk_blk_slices.append((i0,i1,j0,j1))

                shls_slice = (0,mol.nbas,ish0,ish1, jsh0,jsh1,ao_sh0,ao_sh1)
                eri = fint(int2e, mol._atm, mol._bas, mol._env,
                           shls_slice=shls_slice, aosym='s1', ao_loc=ao_loc,
                           cintopt=ao2mopt._cintopt, out=eribuf)
                tmp_xo = lib.einsum('pi,pqrs->iqrs', orbo, eri)
                tmp_xo = lib.einsum('iqrs,sl->ilqr', tmp_xo, orbo_seg)
                bg_save(count, tmp_xo)
                tmp_xo = None
                count += 1
                time1 = log.timer_debug1('partial ao2mo [%d:%d,%d:%d]' %
                                         (ish0,ish1,jsh0,jsh1), *time1)
    eri = eribuf = None
    time1 = time0 = log.timer('mp2 ao2mo_ovov pass1', *time0)

    eris.ovov = eris.feri.create_dataset('ovov', (nocc,nvir,nocc_seg,nvir), 'f8')
    occblk = int(min(nocc, max(BLKMIN, max_memory*.9e6/8/(nao**2*nocc_seg+1)/5)))
    def load(i0, eri):
        if i0 < nocc:
            i1 = min(i0+occblk, nocc)
            for k, (p0,p1,q0,q1) in enumerate(jk_blk_slices):
                eri[:i1-i0,:,p0:p1,q0:q1] = ftmp[str(k)+'a'][i0:i1]
                if p0 != q0:
                    dat = numpy.asarray(ftmp[str(k)+'b'][:,i0:i1])
                    eri[:i1-i0,:,q0:q1,p0:p1] = dat.transpose(1,0,3,2)

    def save(i0, i1, dat):
        eris.ovov[i0:i1] = dat

    buf_prefecth = numpy.empty((occblk,nocc_seg,nao,nao))
    buf = numpy.empty_like(buf_prefecth)
    bufw = numpy.empty((occblk*nocc_seg,nvir**2))
    bufw1 = numpy.empty_like(bufw)
    with lib.call_in_background(load) as prefetch:
        with lib.call_in_background(save) as bsave:
            load(0, buf_prefecth)
            for i0, i1 in lib.prange(0, nocc, occblk):
                buf, buf_prefecth = buf_prefecth, buf
                prefetch(i1, buf_prefecth)
                eri = buf[:i1-i0].reshape((i1-i0)*nocc_seg,nao,nao)

                dat = _ao2mo.nr_e2(eri, orbv, (0,nvir,0,nvir), 's1', 's1', out=bufw)
                bsave(i0, i1, dat.reshape(i1-i0,nocc_seg,nvir,nvir).transpose(0,2,1,3))
                bufw, bufw1 = bufw1, bufw
                time1 = log.timer_debug1('pass2 ao2mo [%d:%d]' % (i0,i1), *time1)

    time0 = log.timer('mp2 ao2mo_ovov pass2', *time0)
    mp._eris = eris
    return eris

def _task_location(n, task=rank):
    ntasks = mpi.pool.size
    seg_size = (n + ntasks - 1) // ntasks
    loc0 = min(n, seg_size * task)
    loc1 = min(n, loc0 + seg_size)
    return loc0, loc1

def _sync_(mp):
    return mp.unpack_(comm.bcast(mp.pack()))

