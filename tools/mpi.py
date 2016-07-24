#!/usr/bin/env python

import sys
import traceback
import numpy
from mpi4py import MPI
from . import mpi_pool
from .mpi_pool import MPIPool

_registry = {}

if 'pool' not in _registry:
    import atexit
    pool = MPIPool(debug=False)
    _registry['pool'] = pool
    atexit.register(pool.close)

comm = pool.comm
rank = pool.rank

def static_partition(tasks):
    size = len(tasks)
    segsize = (size+pool.size-1) // pool.size
    start = pool.rank * segsize
    stop = min(size, start+segsize)
    return tasks[start:stop]

def work_balanced_partition(tasks, costs=None):
    if costs is None:
        costs = numpy.ones(tasks)
    if rank == 0:
        segsize = float(sum(costs)) / pool.size
        loads = []
        cum_costs = numpy.cumsum(costs)
        start_id = 0
        for k in range(pool.size):
            stop_id = numpy.argmin(abs(cum_costs - (k+1)*segsize)) + 1
            stop_id = max(stop_id, start_id+1)
            loads.append([start_id,stop_id])
            start_id = stop_id
        comm.bcast(loads)
    else:
        loads = comm.bcast()
    if rank < len(loads):
        start, stop = loads[rank]
        return tasks[start:stop]
    else:
        return tasks[:0]

def bcast(buf, root=0):
    buf = numpy.asarray(buf, order='C')
    shape, dtype = comm.bcast((buf.shape, buf.dtype))
    if rank != root:
        buf = numpy.empty(shape, dtype=dtype)
    comm.Bcast(buf, root)
    return buf

def reduce(sendbuf, op=MPI.SUM, root=0):
    sendbuf = numpy.asarray(sendbuf, order='C')
    shape, mpi_dtype = comm.bcast((sendbuf.shape, sendbuf.dtype.char))
    _assert(sendbuf.shape == shape and sendbuf.dtype.char == mpi_dtype)

    recvbuf = numpy.zeros_like(sendbuf)
    comm.Reduce(sendbuf, recvbuf, op, root)
    if rank == root:
        return recvbuf
    else:
        return sendbuf

def allreduce(sendbuf, op=MPI.SUM):
    sendbuf = numpy.asarray(sendbuf, order='C')
    shape, mpi_dtype = comm.bcast((sendbuf.shape, sendbuf.dtype.char))
    _assert(sendbuf.shape == shape and sendbuf.dtype.char == mpi_dtype)

    recvbuf = numpy.zeros_like(sendbuf)
    comm.Allreduce(sendbuf, recvbuf, op)
    return recvbuf

def gather(sendbuf, root=0):
#    if pool.debug:
#        if rank == 0:
#            res = [sendbuf]
#            for k in range(1, pool.size):
#                dat = comm.recv(source=k)
#                res.append(dat)
#            return numpy.vstack([x for x in res if len(x) > 0])
#        else:
#            comm.send(sendbuf, dest=0)
#            return sendbuf

    sendbuf = numpy.asarray(sendbuf, order='C')
    mpi_dtype = sendbuf.dtype.char
    if rank == root:
        size_dtype = comm.gather((sendbuf.size, mpi_dtype), root=root)
        _assert(all(x[1] == mpi_dtype for x in size_dtype if x[0] > 0))
        counts = numpy.array([x[0] for x in size_dtype])
        displs = numpy.append([0], numpy.cumsum(counts)[:-1])
        recvbuf = numpy.empty(sum(counts), dtype=sendbuf.dtype)
        comm.Gatherv([sendbuf.ravel(), mpi_dtype],
                     [recvbuf.ravel(), counts, displs, mpi_dtype], root)
        return recvbuf.reshape((-1,) + sendbuf[0].shape)
    else:
        comm.gather((sendbuf.size, mpi_dtype), root=root)
        comm.Gatherv([sendbuf.ravel(), mpi_dtype], None, root)
        return sendbuf

def allgather(sendbuf):
    sendbuf = numpy.asarray(sendbuf, order='C')
    shape, mpi_dtype = comm.bcast((sendbuf.shape, sendbuf.dtype.char))
    _assert(sendbuf.dtype.char == mpi_dtype or sendbuf.size == 0)
    counts = numpy.array(comm.allgather(sendbuf.size))
    displs = numpy.append([0], numpy.cumsum(counts)[:-1])
    recvbuf = numpy.empty(sum(counts), dtype=sendbuf.dtype)
    comm.Allgatherv([sendbuf.ravel(), mpi_dtype],
                    [recvbuf.ravel(), counts, displs, mpi_dtype])
    return recvbuf.reshape((-1,) + shape[1:])

def sendrecv(sendbuf, source=0, dest=0):
    if source == dest:
        return sendbuf

    if rank == source:
        sendbuf = numpy.asarray(sendbuf, order='C')
        comm.send((sendbuf.shape, sendbuf.dtype), dest=dest)
        comm.Send(sendbuf, dest=dest)
        return sendbuf
    elif rank == dest:
        shape, dtype = comm.recv(source=source)
        recvbuf = numpy.empty(shape, dtype=dtype)
        comm.Recv(recvbuf, source=source)
        return recvbuf

def _assert(condition):
    if not condition:
        sys.stderr.write(''.join(traceback.format_stack()[:-1]))
        comm.Abort()

def register_for(obj):
    global _registry
    key = id(obj)
    # Keep track of the object in a global registry.  On slave nodes, the
    # object can be accessed from global registry.
    _registry[key] = obj
    keys = comm.gather(key)
    if rank == 0:
        obj._reg_keys = keys
    return obj

def del_registry(reg_keys):
    if reg_keys:
        def f(reg_keys):
            from mpi4pyscf.tools import mpi
            mpi._registry[reg_keys[mpi.rank]]
        pool.apply(f, reg_keys, reg_keys)
    return []
