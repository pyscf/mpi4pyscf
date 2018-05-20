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
from pyscf.cc import _ccsd

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

    cpu2 = list(cpu1)
    eris = mycc._eris
    fvo = eris.fock[nocc:,:nocc].copy()
    mo_energy = eris.fock.diagonal().copy()
    et_sum = numpy.zeros(1, dtype=t1T.dtype)
    drv = _ccsd.libcc.MPICCsd_t_contract
    def contract(slices, data):
        #vvop_ab, vvop_ac, vvop_ba, vvop_bc, vvop_ca, vvop_cb, \
        #        vooo_a, vooo_b, vooo_c, t2T_a, t2T_b, t2T_c = data
        data_ptrs = [x.ctypes.data_as(ctypes.c_void_p) for x in data]
        data_ptrs = (ctypes.c_void_p*12)(*data_ptrs)
        drv(et_sum.ctypes.data_as(ctypes.c_void_p),
            mo_energy.ctypes.data_as(ctypes.c_void_p),
            t1T.ctypes.data_as(ctypes.c_void_p),
            fvo.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nocc), ctypes.c_int(nvir),
            (ctypes.c_int*6)(*slices), data_ptrs)
        cpu2[:] = log.timer_debug1('contract %d:%d,%d:%d'%(a0,a1,b0,b1), *cpu2)

    with GlobalDataHandler(mycc) as daemon:
        v_seg_ranges = daemon.data_partition
        tasks = []
        for ka, (a0, a1) in enumerate(v_seg_ranges):
            for kb, (b0, b1) in enumerate(v_seg_ranges[:ka+1]):
                for c0, c1 in v_seg_ranges[:kb+1]:
                    tasks.append((a0, a1, b0, b1, c0, c1))

        with lib.call_in_background(contract) as async_contract:
            for task in mpi.work_share_partition(tasks, loadmin=10):
                log.all_debug2('segment_location %s', task)
                data = [None] * 12
                daemon.request_(task, data)
                async_contract(task, data)

    et = comm.allreduce(et_sum[0] * 2).real
    log.timer('CCSD(T)', *cpu0)
    log.note('CCSD(T) correction = %.15g', et)
    return et


BLKMIN = 4

INQUIRY = 50150
TRANSFER_DATA = 50151

class GlobalDataHandler(object):
    def __init__(self, mycc):
        self._cc = mycc
        self.daemon = None

        nocc, nvir = mycc.t1.shape
        nmo = nocc + nvir
        nvir_seg = (nvir + mpi.pool.size - 1) // mpi.pool.size
        max_memory = mycc.max_memory - lib.current_memory()[0]
        blksize = int(max(BLKMIN, (max_memory*.9e6/8/(6*nvir*nocc))**.5 - nocc/4))
        blksize = min(nvir//4+2, nvir_seg, min(comm.allgather(blksize)))
        logger.debug1(mycc, 'GlobalDataHandler blksize %s', blksize)

        self.vranges = []
        self.data_partition = []
        self.segment_location = {}
        for task in range(mpi.pool.size):
            p0 = nvir_seg * task
            p1 = min(nvir, p0 + nvir_seg)
            self.vranges.append((p0, p1))

            for j0, j1 in lib.prange(p0, p1, blksize):
                self.data_partition.append((j0, j1))
                self.segment_location[j0] = task
        logger.debug1(mycc, 'segment_location %s', self.segment_location)

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
            if loc == rank:
                for k, (name, slices) in zip(data_idx, tensors):
                    data[k] = numpy.asarray(self._get_tensor(name, slices), order='C')
            else:
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
        vloc0, vloc1 = self.vranges[rank]
        nvir_seg = vloc1 - vloc0

        max_memory = mycc.max_memory - lib.current_memory()[0]
        blksize = min(nvir, max(16, int(max_memory*.3e6/8/(nvir*nocc*nmo))))
        self.eri_tmp = lib.H5TmpFile()
        vvop = self.eri_tmp.create_dataset('vvop', (nvir_seg,nvir,nocc,nmo), 'f8')
        with lib.call_in_background(vvop.__setitem__) as save:
            for p0, p1 in mpi.prange(vloc0, vloc1, blksize):
                j0, j1 = p0 - vloc0, p1 - vloc0
                sub_locs = comm.allgather((p0,p1))
                vvvo = mpi.alltoall([eris.vvvo[:,:,q0:q1] for q0, q1 in sub_locs],
                                    split_recvbuf=True)

                buf = numpy.empty((p1-p0,nvir,nocc,nmo), dtype=t2T.dtype)
                buf[:,:,:,:nocc] = eris.ovov[:,j0:j1].conj().transpose(1,3,0,2)
                for k, (q0, q1) in enumerate(self.vranges):
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
                        else:
                            mpi.send(self._get_tensor(task, slices), dest,
                                     tag=TRANSFER_DATA)
                time.sleep(interval)

        daemon = threading.Thread(target=send_data)
        daemon.start()
        return daemon

    def _get_tensor(self, name, slices):
        vloc0, vloc1 = self.vranges[rank]
        if len(slices) == 4:
            slices = (slice(slices[0]-vloc0, slices[1]-vloc0),
                      slice(*slices[2:4]))
        else:
            slices = slice(slices[0]-vloc0, slices[1]-vloc0)

        if name == 'vvop':
            tensor = self.eri_tmp['vvop'][slices]
        elif name == 't2':
            t2T = self._cc.t2.transpose(2,3,1,0)
            tensor = t2T[slices]
        elif name == 'vooo':
            tensor = self._cc._eris.ovoo[:,slices].transpose(1,0,3,2).conj()
        else:
            raise RuntimeError('Unknown tensor %s' % name)
        return tensor

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
        self.eri_tmp = None


