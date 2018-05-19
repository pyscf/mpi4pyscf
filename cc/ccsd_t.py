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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
RHF-CCSD(T) for real integrals
'''

import time
import ctypes
import threading
import numpy
from pyscf import lib

from mpi4py import MPI
from mpi4pyscf.lib import logger
from mpi4pyscf.tools import mpi

comm = mpi.comm
rank = mpi.rank


@mpi.parallel_call
def kernel(mycc):
    cpu1 = cpu0 = (time.clock(), time.time())
    log = logger.new_logger(mycc)

    t1T = mycc.t1.T
    nvir, nocc = t1T.shape
    nmo = nocc + nvir

    eris = mycc._eris
    fvo = eris.fock[nocc:,:nocc]
    mo_e = eris.fock.diagonal()
    e_occ, e_vir = mo_e[:nocc], mo_e[nocc:]
    eijk = lib.direct_sum('i,j,k->ijk', e_occ, e_occ, e_occ)

    def get_w(a, b, c, a0, b0, c0, vvop, vooo, t2Tb, t2Tc):
        w = numpy.einsum('if,fkj->ijk', vvop[a-a0,b-b0,:,nocc:], t2Tc[c-c0,:])
        w-= numpy.einsum('ijm,mk->ijk', vooo[a-a0], t2Tb[b-b0,c])
        return w
    def get_v(a, b, c, a0, b0, c0, vvop, t2T):
        v = numpy.einsum('ij,k->ijk', vvop[a-a0,b-b0,:,:nocc], t1T[c])
        v+= numpy.einsum('ij,k->ijk', t2T[a-a0,b], fvo[c])
        return v
    def r3(w):
        return (4 * w + w.transpose(1,2,0) + w.transpose(2,0,1)
                - 2 * w.transpose(2,1,0) - 2 * w.transpose(0,2,1)
                - 2 * w.transpose(1,0,2))

    et_sum = [0]
    def contract(slices, data):
        a0, a1, b0, b1, c0, c1 = slices
        vvop_ab, vvop_ac, vvop_ba, vvop_bc, vvop_ca, vvop_cb, \
                vooo_a, vooo_b, vooo_c, t2T_a, t2T_b, t2T_c = data
        et = 0
        for a in range(a0, a1):
            for b in range(b0, min(a+1, b1)):
                for c in range(c0, min(b+1, c1)):
                    d3 = eijk - e_vir[a] - e_vir[b] - e_vir[c]
                    if a == c:  # a == b == c
                        d3 *= 6
                    elif a == b or b == c:
                        d3 *= 2

                    wabc = get_w(a, b, c, a0, b0, c0, vvop_ab, vooo_a, t2T_b, t2T_c)
                    wacb = get_w(a, c, b, a0, c0, b0, vvop_ac, vooo_a, t2T_c, t2T_b)
                    wbac = get_w(b, a, c, b0, a0, c0, vvop_ba, vooo_b, t2T_a, t2T_c)
                    wbca = get_w(b, c, a, b0, c0, a0, vvop_bc, vooo_b, t2T_c, t2T_a)
                    wcab = get_w(c, a, b, c0, a0, b0, vvop_ca, vooo_c, t2T_a, t2T_b)
                    wcba = get_w(c, b, a, c0, b0, a0, vvop_cb, vooo_c, t2T_b, t2T_a)
                    vabc = get_v(a, b, c, a0, b0, c0, vvop_ab, t2T_a)
                    vacb = get_v(a, c, b, a0, c0, b0, vvop_ac, t2T_a)
                    vbac = get_v(b, a, c, b0, a0, c0, vvop_ba, t2T_b)
                    vbca = get_v(b, c, a, b0, c0, a0, vvop_bc, t2T_b)
                    vcab = get_v(c, a, b, c0, a0, b0, vvop_ca, t2T_c)
                    vcba = get_v(c, b, a, c0, b0, a0, vvop_cb, t2T_c)
                    zabc = r3(wabc + .5 * vabc) / d3
                    zacb = r3(wacb + .5 * vacb) / d3
                    zbac = r3(wbac + .5 * vbac) / d3
                    zbca = r3(wbca + .5 * vbca) / d3
                    zcab = r3(wcab + .5 * vcab) / d3
                    zcba = r3(wcba + .5 * vcba) / d3

                    et+= numpy.einsum('ijk,ijk', wabc, zabc.conj())
                    et+= numpy.einsum('ikj,ijk', wacb, zabc.conj())
                    et+= numpy.einsum('jik,ijk', wbac, zabc.conj())
                    et+= numpy.einsum('jki,ijk', wbca, zabc.conj())
                    et+= numpy.einsum('kij,ijk', wcab, zabc.conj())
                    et+= numpy.einsum('kji,ijk', wcba, zabc.conj())

                    et+= numpy.einsum('ijk,ijk', wacb, zacb.conj())
                    et+= numpy.einsum('ikj,ijk', wabc, zacb.conj())
                    et+= numpy.einsum('jik,ijk', wcab, zacb.conj())
                    et+= numpy.einsum('jki,ijk', wcba, zacb.conj())
                    et+= numpy.einsum('kij,ijk', wbac, zacb.conj())
                    et+= numpy.einsum('kji,ijk', wbca, zacb.conj())

                    et+= numpy.einsum('ijk,ijk', wbac, zbac.conj())
                    et+= numpy.einsum('ikj,ijk', wbca, zbac.conj())
                    et+= numpy.einsum('jik,ijk', wabc, zbac.conj())
                    et+= numpy.einsum('jki,ijk', wacb, zbac.conj())
                    et+= numpy.einsum('kij,ijk', wcba, zbac.conj())
                    et+= numpy.einsum('kji,ijk', wcab, zbac.conj())

                    et+= numpy.einsum('ijk,ijk', wbca, zbca.conj())
                    et+= numpy.einsum('ikj,ijk', wbac, zbca.conj())
                    et+= numpy.einsum('jik,ijk', wcba, zbca.conj())
                    et+= numpy.einsum('jki,ijk', wcab, zbca.conj())
                    et+= numpy.einsum('kij,ijk', wabc, zbca.conj())
                    et+= numpy.einsum('kji,ijk', wacb, zbca.conj())

                    et+= numpy.einsum('ijk,ijk', wcab, zcab.conj())
                    et+= numpy.einsum('ikj,ijk', wcba, zcab.conj())
                    et+= numpy.einsum('jik,ijk', wacb, zcab.conj())
                    et+= numpy.einsum('jki,ijk', wabc, zcab.conj())
                    et+= numpy.einsum('kij,ijk', wbca, zcab.conj())
                    et+= numpy.einsum('kji,ijk', wbac, zcab.conj())

                    et+= numpy.einsum('ijk,ijk', wcba, zcba.conj())
                    et+= numpy.einsum('ikj,ijk', wcab, zcba.conj())
                    et+= numpy.einsum('jik,ijk', wbca, zcba.conj())
                    et+= numpy.einsum('jki,ijk', wbac, zcba.conj())
                    et+= numpy.einsum('kij,ijk', wacb, zcba.conj())
                    et+= numpy.einsum('kji,ijk', wabc, zcba.conj())
        et_sum[0] += et * 2

    with GlobalDataHandler(mycc) as daemon:
        v_seg_ranges = daemon.data_partition
        tasks = []
        for ka, (a0, a1) in enumerate(v_seg_ranges):
            for kb, (b0, b1) in enumerate(v_seg_ranges[:ka+1]):
                for c0, c1 in v_seg_ranges[:kb+1]:
                    tasks.append((a0, a1, b0, b1, c0, c1))

        with lib.call_in_background(contract) as async_contract:
            for task in mpi.work_share_partition(tasks, loadmin=10):
                data = [None] * 12
                daemon.request_(task, data)
                async_contract(task, data)

    et = comm.allreduce(et_sum[0]).real
    log.timer('CCSD(T)', *cpu0)
    log.note('CCSD(T) correction = %.15g', et)
    return et


BLKMIN = 4

INQUIRY = 50150
TRANSFER_DATA = 50151

class GlobalDataHandler(object):
    def __init__(self, mycc):
        self._cc = mycc
        self.data_partition = None
        self.segment_location = None
        self.daemon = None

    def request_(self, slices, data):
        assert(len(data) == 12)
        job_slots = {}
        def add_job(data_idx, tensor_description):
            slices = tensor_description[1]
            loc = self.segment_location[slices[0]]
            if loc in job_slots:
                indices, tensors = job_slots[loc]
                indices.append(data_idx)
                tensors.append(tensor_description)
                job_slots[loc] = (indices, tensors)
            else:
                job_slots[loc] = ([data_idx], [tensor_description])

        a0, a1, b0, b1, c0, c1 = slices
        add_job(0 , ('vvop', (a0,a1, b0,b1)))
        add_job(1 , ('vvop', (a0,a1, c0,c1)))
        add_job(2 , ('vvop', (b0,b1, a0,a1)))
        add_job(3 , ('vvop', (b0,b1, c0,c1)))
        add_job(4 , ('vvop', (c0,c1, a0,a1)))
        add_job(5 , ('vvop', (c0,c1, b0,b1)))
        add_job(6 , ('vooo', (a0,a1)))
        add_job(7 , ('vooo', (b0,b1)))
        add_job(8 , ('vooo', (c0,c1)))
        add_job(9 , ('t2'  , (a0,a1)))
        add_job(10, ('t2'  , (b0,b1)))
        add_job(11, ('t2'  , (c0,c1)))

        def get(data_idx, tensors, loc):
            comm.send((tensors, rank), dest=loc, tag=INQUIRY)
            for k in data_idx:
                data[k] = mpi.recv(loc, tag=TRANSFER_DATA)

        threads = []
        for loc in job_slots:
            data_idx, tensors = job_slots[loc]
            p = threading.Thread(target=get, args=(data_idx, tensors, loc))
            p.start()
            threads.append(p)
        for p in threads:
            p.join()

    def start(self, interval=0.02):
        mycc = self._cc
        log = logger.new_logger(mycc)
        cpu1 = (time.clock(), time.time())
        eris = mycc._eris
        t2T = mycc.t2.transpose(2,3,0,1)

        nocc, nvir = mycc.t1.shape
        nmo = nocc + nvir
        nvir_seg = (nvir + mpi.pool.size - 1) // mpi.pool.size

        max_memory = mycc.max_memory - lib.current_memory()[0]
        blksize = int(max(BLKMIN, (max_memory*.9e6/8/(6*nvir*nocc))**.5 - nocc/4))
        blksize = min(nvir//4+2, nvir_seg, min(comm.allgather(blksize)))

        vlocs = []
        self.data_partition = []
        self.segment_location = {}
        for task in range(mpi.pool.size):
            p0 = nvir_seg * task
            p1 = min(nvir, p0 + nvir_seg)
            vlocs.append((p0, p1))

            for j0, j1 in lib.prange(p0, p1, blksize):
                self.data_partition.append((j0, j1))
                self.segment_location[j0] = task
        log.debug1('segment_location %s', self.segment_location)

        vloc0, vloc1 = vlocs[rank]
        blksize = min(nvir, max(16, int(max_memory*.3e6/8/(nvir*nocc*nmo))))
        self.ftmp = lib.H5TmpFile()
        vvop = self.ftmp.create_dataset('vvop', (nvir_seg,nvir,nocc,nmo), 'f8')
        with lib.call_in_background(vvop.__setitem__) as save:
            for p0, p1 in mpi.prange(vloc0, vloc1, blksize):
                j0, j1 = p0 - vloc0, p1 - vloc0
                sub_locs = comm.allgather((p0,p1))
                vvvo = mpi.alltoall([eris.vvvo[:,:,q0:q1] for q0, q1 in sub_locs],
                                    split_recvbuf=True)

                buf = numpy.empty((p1-p0,nvir,nocc,nmo), dtype=t2T.dtype)
                buf[:,:,:,:nocc] = eris.ovov[:,j0:j1].conj().transpose(1,3,0,2)
                for k, (q0, q1) in enumerate(vlocs):
                    blk = vvvo[k].reshape(q1-q0,nvir,p1-p0,nocc)
                    buf[:,q0:q1,:,nocc:] = blk.transpose(2,0,3,1)
                save(slice(j0,j1,None), buf)
                buf = vvvo = sub_locs = blk = None
                cpu1 = log.timer_debug1('transpose %d:%d'%(p0,p1), *cpu1)

        def send_data():
            while True:
                while comm.Iprobe(source=MPI.ANY_SOURCE, tag=INQUIRY):
                    tensors, dest = comm.recv(source=MPI.ANY_SOURCE, tag=INQUIRY)
                    for task, slices in tensors:
                        if task == 'Done':
                            return
                        if len(slices) == 4:
                            slices = (slice(slices[0]-vloc0, slices[1]-vloc0),
                                      slice(*slices[2:4]))
                        else:
                            slices = slice(slices[0]-vloc0, slices[1]-vloc0)
                        if task == 'vvop':
                            mpi.send(vvop[slices], dest, tag=TRANSFER_DATA)
                        elif task == 't2':
                            mpi.send(t2T[slices], dest, tag=TRANSFER_DATA)
                        elif task == 'vooo':
                            mpi.send(eris.ovoo[:,slices].transpose(1,0,3,2).conj(),
                                     dest, tag=TRANSFER_DATA)
                        else:
                            raise RuntimeError('Unknown task')
                time.sleep(interval)

        daemon = threading.Thread(target=send_data)
        daemon.start()
        return daemon

    def close(self):
        if rank == 0:
            for i in range(mpi.pool.size):
                comm.send(([('Done', None)], None), dest=i, tag=INQUIRY)
        self.daemon.join()

    def __enter__(self):
        self.daemon = self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        self.ftmp = None


