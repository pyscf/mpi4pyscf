#!/usr/bin/env python

import sys
import time
import enum
import threading
import traceback
import numpy
from mpi4py import MPI
from . import mpi_pool
from .mpi_pool import MPIPool
from pyscf import lib

_registry = {}

if 'pool' not in _registry:
    import atexit
    pool = MPIPool(debug=False)
    _registry['pool'] = pool
    atexit.register(pool.close)

comm = pool.comm
rank = pool.rank
INT_MAX = 2147483647
BLKSIZE = INT_MAX // 32 + 1

def static_partition(tasks):
    size = len(tasks)
    segsize = (size+pool.size-1) // pool.size
    start = pool.rank * segsize
    stop = min(size, start+segsize)
    return tasks[start:stop]

def work_balanced_partition(tasks, costs=None):
    if costs is None:
        costs = numpy.ones(len(tasks))
    if rank == 0:
        cum = numpy.append(0, numpy.cumsum(costs))
        segsize = float(cum[-1]) / (pool.size-.5)
        displs = lib.misc._blocksize_partition(cum, segsize)
        if len(displs) != pool.size+1:
            displs = lib.misc._balanced_partition(cum, pool.size)
        loads = list(zip(displs[:-1], displs[1:]))
        comm.bcast(loads)
    else:
        loads = comm.bcast(None)
    if rank < len(loads):
        start, stop = loads[rank]
        return tasks[start:stop]
    else:
        return tasks[:0]

def work_share_partition(tasks, interval=.1, loadmin=2):
    if pool.size <= 1:
        for task in tasks:
            yield task
        return

    loadmin = min(loadmin, (len(tasks)+pool.size-1)//pool.size)
    loadmin = max(loadmin, len(tasks)//50//pool.size, 1)
    if rank == 0:
        rest_tasks = list(tasks[loadmin*pool.size:][::-1])

    tasks = list(tasks[loadmin*rank:loadmin*rank+loadmin][::-1])
    def distribute_task():
        while True:
            load = comm.gather(len(tasks))
            if rank == 0:
                if rest_tasks:
                    jobs = [None] * pool.size
                    for i in range(pool.size):
                        if rest_tasks and load[i] < loadmin:
                            jobs[i] = rest_tasks.pop()
                else:
                    jobs = [Message.OutOfTasks] * pool.size
                task = comm.scatter(jobs)
            else:
                task = comm.scatter(None)

            if task is not None:
                tasks.insert(0, task)
                if task is Message.OutOfTasks:
                    return

            time.sleep(interval)

    tasks_handler = threading.Thread(target=distribute_task)
    tasks_handler.start()

    while True:
        if tasks:
            task = tasks.pop()
            if task is Message.OutOfTasks:
                break
            yield task
    tasks_handler.join()

def work_stealing_partition(tasks, interval=.1):
    if pool.size <= 1:
        for task in tasks:
            yield task
        return

    tasks = list(static_partition(tasks)[::-1])
    def task_daemon():
        while True:
            time.sleep(interval)

            ntasks = len(tasks)
            loads = comm.allgather(ntasks)
            if all(n <= 1 for n in loads):
                tasks.insert(0, Message.OutOfTasks)
                break

            elif any(n <= 1 for n in loads) and any(n >= 3 for n in loads):
                loads = numpy.array(loads)
                ntasks_mean = int(loads.mean()) + 1
                mpi_size = pool.size
                # number of tasks may be changed when reaching this spot. It
                # may lead to the condition that len(tasks) is less than the
                # ntasks-ntasks_mean and an error "pop from empty list". The
                # status of tasks should be checked before calling tasks.pop
                to_send = [tasks.pop() for i in range(len(tasks)-ntasks_mean) if tasks]

                to_distriubte = comm.gather(to_send)
                if rank == 0:
                    to_distriubte = lib.flatten(to_distriubte)
                    to_append = [[] for i in range(mpi_size)]
                    i = 1
                    while to_distriubte and i < mpi_size:
                        npop = ntasks_mean - loads[i]
                        to_append[i] = to_distriubte[:npop]
                        to_distriubte = to_distriubte[npop:]
                        i += 1
                    to_append[0] = to_distriubte
                else:
                    to_append = None

                to_append = comm.scatter(to_append)
                for task in to_append:
                    tasks.insert(0, task)

    tasks_handler = threading.Thread(target=task_daemon)
    tasks_handler.start()

    while True:
        if tasks:
            task = tasks.pop()
            if task is Message.OutOfTasks:
                break
            yield task

    tasks_handler.join()

def _create_dtype(dat):
    mpi_dtype = MPI._typedict[dat.dtype.char]
    # the smallest power of 2 greater than dat.size/INT_MAX
    deriv_dtype_len = 1 << max(0, dat.size.bit_length()-31)
    deriv_dtype = mpi_dtype.Create_contiguous(deriv_dtype_len).Commit()
    count, rest = dat.size.__divmod__(deriv_dtype_len)
    return deriv_dtype, count, rest

def bcast_test(buf, root=0):  # To test, maybe with better performance
    buf = numpy.asarray(buf, order='C')
    shape, dtype = comm.bcast((buf.shape, buf.dtype.char))
    if rank != root:
        buf = numpy.empty(shape, dtype=dtype)

    if buf.size <= BLKSIZE:
        comm.Bcast([buf, buf.dtype.char], root)
    else:
        deriv_dtype, count, rest = _create_dtype(buf)
        comm.Bcast([buf, count, deriv_dtype], root)
        comm.Bcast([buf[-rest*deriv_dtype.size:], deriv_dtype], root)
    return buf

def bcast(buf, root=0):
    buf = numpy.asarray(buf, order='C')
    shape, dtype = comm.bcast((buf.shape, buf.dtype.char))
    if rank != root:
        buf = numpy.empty(shape, dtype=dtype)

    dtype = buf.dtype.char
    buf_seg = numpy.ndarray(buf.size, dtype=buf.dtype, buffer=buf)
    for p0, p1 in lib.prange(0, buf.size, BLKSIZE):
        comm.Bcast([buf_seg[p0:p1], dtype], root)
    return buf


def bcast_tagged_array(arr):
    '''Broadcast big nparray or tagged array.'''
    if comm.bcast(not isinstance(arr, numpy.ndarray)):
        return comm.bcast(arr)

    new_arr = bcast(arr)

    if comm.bcast(isinstance(arr, lib.NPArrayWithTag)):
        new_arr = lib.tag_array(new_arr)
        if rank == 0:
            kv = []
            for k, v in arr.__dict__.items():
                if isinstance(v, numpy.ndarray) and v.nbytes > 1e5:
                    kv.append((k, Message.NparrayToBcast))
                else:
                    kv.append((k, v))
            comm.bcast(kv)
        else:
            kv = comm.bcast(None)
            new_arr.__dict__.update(kv)

        for k, v in kv:
            if v is Message.NparrayToBcast:
                new_arr.k = bcast(v)

    if rank != 0:
        arr = new_arr
    return arr


def reduce(sendbuf, op=MPI.SUM, root=0):
    sendbuf = numpy.asarray(sendbuf, order='C')
    shape, mpi_dtype = comm.bcast((sendbuf.shape, sendbuf.dtype.char))
    _assert(sendbuf.shape == shape and sendbuf.dtype.char == mpi_dtype)

    dtype = sendbuf.dtype.char
    recvbuf = numpy.zeros_like(sendbuf)
    send_seg = numpy.ndarray(sendbuf.size, dtype=sendbuf.dtype, buffer=sendbuf)
    recv_seg = numpy.ndarray(recvbuf.size, dtype=recvbuf.dtype, buffer=recvbuf)
    for p0, p1 in lib.prange(0, sendbuf.size, BLKSIZE):
        comm.Reduce([send_seg[p0:p1], dtype],
                    [recv_seg[p0:p1], dtype], op, root)

    if rank == root:
        return recvbuf
    else:
        return sendbuf

def allreduce(sendbuf, op=MPI.SUM):
    sendbuf = numpy.asarray(sendbuf, order='C')
    shape, mpi_dtype = comm.bcast((sendbuf.shape, sendbuf.dtype.char))
    _assert(sendbuf.shape == shape and sendbuf.dtype.char == mpi_dtype)

    dtype = sendbuf.dtype.char
    recvbuf = numpy.zeros_like(sendbuf)
    send_seg = numpy.ndarray(sendbuf.size, dtype=sendbuf.dtype, buffer=sendbuf)
    recv_seg = numpy.ndarray(recvbuf.size, dtype=recvbuf.dtype, buffer=recvbuf)
    for p0, p1 in lib.prange(0, sendbuf.size, BLKSIZE):
        comm.Allreduce([send_seg[p0:p1], dtype],
                       [recv_seg[p0:p1], dtype], op)
    return recvbuf

def scatter(sendbuf, root=0):
    if rank == root:
        mpi_dtype = numpy.result_type(*sendbuf).char
        shape = comm.scatter([x.shape for x in sendbuf])
        counts = numpy.asarray([x.size for x in sendbuf])
        comm.bcast((mpi_dtype, counts))
        sendbuf = [numpy.asarray(x, mpi_dtype).ravel() for x in sendbuf]
        sendbuf = numpy.hstack(sendbuf)
    else:
        shape = comm.scatter(None)
        mpi_dtype, counts = comm.bcast(None)

    displs = numpy.append(0, numpy.cumsum(counts[:-1]))
    recvbuf = numpy.empty(numpy.prod(shape), dtype=mpi_dtype)

    #DONOT use lib.prange. lib.prange may terminate early in some processes
    for p0, p1 in prange(0, numpy.max(counts), BLKSIZE):
        counts_seg = _segment_counts(counts, p0, p1)
        comm.Scatterv([sendbuf, counts_seg, displs+p0, mpi_dtype],
                      [recvbuf[p0:p1], mpi_dtype], root)
    return recvbuf.reshape(shape)

def gather(sendbuf, root=0, split_recvbuf=False):
    #if pool.debug:
    #    if rank == 0:
    #        res = [sendbuf]
    #        for k in range(1, pool.size):
    #            dat = comm.recv(source=k)
    #            res.append(dat)
    #        return numpy.vstack([x for x in res if len(x) > 0])
    #    else:
    #        comm.send(sendbuf, dest=0)
    #        return sendbuf

    sendbuf = numpy.asarray(sendbuf, order='C')
    shape = sendbuf.shape
    size_dtype = comm.allgather((shape, sendbuf.dtype.char))
    rshape = [x[0] for x in size_dtype]
    counts = numpy.array([numpy.prod(x) for x in rshape])

    mpi_dtype = numpy.result_type(*[x[1] for x in size_dtype]).char
    _assert(sendbuf.dtype == mpi_dtype or sendbuf.size == 0)

    if rank == root:
        displs = numpy.append(0, numpy.cumsum(counts[:-1]))
        recvbuf = numpy.empty(sum(counts), dtype=mpi_dtype)

        sendbuf = sendbuf.ravel()
        for p0, p1 in lib.prange(0, numpy.max(counts), BLKSIZE):
            counts_seg = _segment_counts(counts, p0, p1)
            comm.Gatherv([sendbuf[p0:p1], mpi_dtype],
                         [recvbuf, counts_seg, displs+p0, mpi_dtype], root)
        if split_recvbuf:
            return [recvbuf[p0:p0+c].reshape(shape)
                    for p0,c,shape in zip(displs,counts,rshape)]
        else:
            try:
                return recvbuf.reshape((-1,) + shape[1:])
            except ValueError:
                return recvbuf
    else:
        send_seg = sendbuf.ravel()
        for p0, p1 in lib.prange(0, numpy.max(counts), BLKSIZE):
            comm.Gatherv([send_seg[p0:p1], mpi_dtype], None, root)
        return sendbuf


def allgather(sendbuf, split_recvbuf=False):
    sendbuf = numpy.asarray(sendbuf, order='C')
    shape = sendbuf.shape
    attr = comm.allgather((shape, sendbuf.dtype.char))
    rshape = [x[0] for x in attr]
    counts = numpy.array([numpy.prod(x) for x in rshape])
    mpi_dtype = numpy.result_type(*[x[1] for x in attr]).char
    _assert(sendbuf.dtype.char == mpi_dtype or sendbuf.size == 0)

    displs = numpy.append(0, numpy.cumsum(counts[:-1]))
    recvbuf = numpy.empty(sum(counts), dtype=mpi_dtype)

    sendbuf = sendbuf.ravel()
    for p0, p1 in lib.prange(0, numpy.max(counts), BLKSIZE):
        counts_seg = _segment_counts(counts, p0, p1)
        comm.Allgatherv([sendbuf[p0:p1], mpi_dtype],
                        [recvbuf, counts_seg, displs+p0, mpi_dtype])
    if split_recvbuf:
        return [recvbuf[p0:p0+c].reshape(shape)
                for p0,c,shape in zip(displs,counts,rshape)]
    else:
        try:
            return recvbuf.reshape((-1,) + shape[1:])
        except ValueError:
            return recvbuf

def alltoall(sendbuf, split_recvbuf=False):
    if isinstance(sendbuf, numpy.ndarray):
        mpi_dtype = comm.bcast(sendbuf.dtype.char)
        sendbuf = numpy.asarray(sendbuf, mpi_dtype, 'C')
        nrow = sendbuf.shape[0]
        ncol = sendbuf.size // nrow
        segsize = (nrow+pool.size-1) // pool.size * ncol
        sdispls = numpy.arange(0, pool.size*segsize, segsize)
        sdispls[sdispls>sendbuf.size] = sendbuf.size
        scounts = numpy.append(sdispls[1:]-sdispls[:-1], sendbuf.size-sdispls[-1])
        rshape = comm.alltoall(scounts)
    else:
        _assert(len(sendbuf) == pool.size)
        mpi_dtype = comm.bcast(sendbuf[0].dtype.char)
        sendbuf = [numpy.asarray(x, mpi_dtype) for x in sendbuf]
        rshape = comm.alltoall([x.shape for x in sendbuf])
        scounts = numpy.asarray([x.size for x in sendbuf])
        sdispls = numpy.append(0, numpy.cumsum(scounts[:-1]))
        sendbuf = numpy.hstack([x.ravel() for x in sendbuf])

    rcounts = numpy.asarray([numpy.prod(x) for x in rshape])
    rdispls = numpy.append(0, numpy.cumsum(rcounts[:-1]))
    recvbuf = numpy.empty(sum(rcounts), dtype=mpi_dtype)

    max_counts = max(numpy.max(scounts), numpy.max(rcounts))
    sendbuf = sendbuf.ravel()
    #DONOT use lib.prange. lib.prange may terminate early in some processes
    for p0, p1 in prange(0, max_counts, BLKSIZE):
        scounts_seg = _segment_counts(scounts, p0, p1)
        rcounts_seg = _segment_counts(rcounts, p0, p1)
        comm.Alltoallv([sendbuf, scounts_seg, sdispls+p0, mpi_dtype],
                       [recvbuf, rcounts_seg, rdispls+p0, mpi_dtype])

    if split_recvbuf:
        return [recvbuf[p0:p0+c].reshape(shape)
                for p0,c,shape in zip(rdispls, rcounts, rshape)]
    else:
        return recvbuf

def send(sendbuf, dest=0, tag=0):
    sendbuf = numpy.asarray(sendbuf, order='C')
    dtype = sendbuf.dtype.char
    comm.send((sendbuf.shape, dtype), dest=dest, tag=tag)
    send_seg = numpy.ndarray(sendbuf.size, dtype=sendbuf.dtype, buffer=sendbuf)
    for p0, p1 in lib.prange(0, sendbuf.size, BLKSIZE):
        comm.Send([send_seg[p0:p1], dtype], dest=dest, tag=tag)
    return sendbuf

def recv(source=0, tag=0):
    shape, dtype = comm.recv(source=source, tag=tag)
    recvbuf = numpy.empty(shape, dtype=dtype)
    recv_seg = numpy.ndarray(recvbuf.size, dtype=recvbuf.dtype, buffer=recvbuf)
    for p0, p1 in lib.prange(0, recvbuf.size, BLKSIZE):
        comm.Recv([recv_seg[p0:p1], dtype], source=source, tag=tag)
    return recvbuf

def sendrecv(sendbuf, source=0, dest=0, tag=0):
    if source == dest:
        return sendbuf

    if rank == source:
        send(sendbuf, dest, tag)
    elif rank == dest:
        return recv(source, tag)

def rotate(sendbuf, blocking=True, tag=0):
    '''On every process, pass the sendbuf to the next process.
    Node-ID  Before-rotate  After-rotate
    node-0   buf-0          buf-1
    node-1   buf-1          buf-2
    node-2   buf-2          buf-3
    node-3   buf-3          buf-0
    '''
    if pool.size <= 1:
        return sendbuf

    if rank == 0:
        prev_node = pool.size - 1
        next_node = 1
    elif rank == pool.size - 1:
        prev_node = rank - 1
        next_node = 0
    else:
        prev_node = rank - 1
        next_node = rank + 1

    if isinstance(sendbuf, numpy.ndarray):
        if blocking:
            if rank % 2 == 0:
                send(sendbuf, prev_node, tag)
                recvbuf = recv(next_node, tag)
            else:
                recvbuf = recv(next_node, tag)
                send(sendbuf, prev_node, tag)
        else:
            handler = lib.ThreadWithTraceBack(target=send, args=(sendbuf, prev_node, tag))
            handler.start()
            recvbuf = recv(next_node, tag)
            handler.join()
    else:
        if rank % 2 == 0:
            comm.send(sendbuf, dest=next_node, tag=tag)
            recvbuf = comm.recv(source=prev_node, tag=tag)
        else:
            recvbuf = comm.recv(source=prev_node, tag=tag)
            comm.send(sendbuf, dest=next_node, tag=tag)
    return recvbuf

def _assert(condition):
    if not condition:
        sys.stderr.write(''.join(traceback.format_stack()[:-1]))
        comm.Abort()

def del_registry(reg_procs):
    if reg_procs:
        def f(reg_procs):
            from mpi4pyscf.tools import mpi
            mpi._registry.pop(reg_procs[mpi.rank])
        pool.apply(f, (reg_procs,), (reg_procs,))
    return []

def _init_on_workers(module, name, args, kwargs):
    import importlib
    from pyscf.gto import mole
    from pyscf.pbc.gto import cell
    from mpi4pyscf.tools import mpi
    if args is None and kwargs is None:  # Not to call __init__ function on workers
        if module is None:  # master proccess
            obj = name
            if hasattr(obj, 'cell'):
                mol_str = obj.cell.dumps()
            elif hasattr(obj, 'mol'):
                mol_str = obj.mol.dumps()
            else:
                mol_str = None
            mpi.comm.bcast((mol_str, obj.pack()))
        else:
            cls = getattr(importlib.import_module(module), name)
            obj = cls.__new__(cls)
            mol_str, obj_attr = mpi.comm.bcast(None)
            obj.unpack_(obj_attr)
            if mol_str is not None:
                if '"ke_cutoff"' in mol_str:
                    obj.cell = cell.loads(mol_str)
                elif '_bas' in mol_str:
                    obj.mol = mole.loads(mol_str)

    elif module is None:  # master proccess
        obj = name

    else:
        # Guess whether the args[0] is the serialized mole or cell objects
        if isinstance(args[0], str) and args[0][0] == '{':
            if '"ke_cutoff"' in args[0]:
                c = cell.loads(args[0])
                args = (c,) + args[1:]
            elif '_bas' in args[0]:
                m = mole.loads(args[0])
                args = (m,) + args[1:]
        cls = getattr(importlib.import_module(module), name)
        obj = cls(*args, **kwargs)

    key = id(obj)
    mpi._registry[key] = obj
    regs = mpi.comm.gather(key)
    return regs

if rank == 0:
    from pyscf.gto import mole
    def _init_and_register(cls, with__init__=True):
        old_init = cls.__init__
        def init(obj, *args, **kwargs):
            old_init(obj, *args, **kwargs)

# * Do not issue mpi communication inside the __init__ method
# * Avoid the distributed class being called twice from subclass  __init__ method
# * hasattr(obj, '_reg_procs') to ensure class is created only once
# * If class initialized in mpi session, bypass the distributing step
            if pool.worker_status == 'P' and not hasattr(obj, '_reg_procs'):
                cls = obj.__class__
                if len(args) > 0 and isinstance(args[0], mole.Mole):
                    regs = pool.apply(_init_on_workers, (None, obj, args, kwargs),
                                      (cls.__module__, cls.__name__,
                                       (args[0].dumps(),)+args[1:], kwargs))
                elif with__init__:
                    regs = pool.apply(_init_on_workers, (None, obj, args, kwargs),
                                      (cls.__module__, cls.__name__, args, kwargs))
                else:
                    regs = pool.apply(_init_on_workers, (None, obj, None, None),
                                      (cls.__module__, cls.__name__, None, None))

                # Keep track of the object in a global registry.  The object can
                # be accessed from global registry on workers.
                obj._reg_procs = regs
        return init
    def _with_enter(obj):
        return obj
    def _with_exit(obj):
        obj._reg_procs = del_registry(obj._reg_procs)

    def register_class(cls):
        cls.__init__ = _init_and_register(cls)
        cls.__enter__ = _with_enter
        cls.__exit__ = _with_exit
        cls.close = _with_exit
        return cls

    def register_class_without__init__(cls):
        cls.__init__ = _init_and_register(cls, False)
        cls.__enter__ = _with_enter
        cls.__exit__ = _with_exit
        cls.close = _with_exit
        return cls
else:
    def register_class(cls):
        return cls
    def register_class_without__init__(cls):
        return cls

def _distribute_call(module, name, reg_procs, args, kwargs):
    import importlib
    dev = reg_procs
    if module is None:  # Master process
        fn = name
    else:
        fn = getattr(importlib.import_module(module), name)
        if dev is None:
            pass
        elif isinstance(dev, str) and dev[0] == '{':
            # Guess whether dev is Mole or Cell, then deserialize dev
            from pyscf.gto import mole
            from pyscf.pbc.gto import cell
            if '"ke_cutoff"' in dev:
                dev = cell.loads(dev)
            elif '_bas' in dev:
                dev = mole.loads(dev)
        else:
            from mpi4pyscf.tools import mpi
            dev = mpi._registry[reg_procs[mpi.rank]]
    return fn(dev, *args, **kwargs)

if rank == 0:
    def parallel_call(fn=None, skip_args=None, skip_kwargs=None):
        '''
        Kwargs:
            skip_args (list of ints): the argument indices in the args list.
                The arguments specified in skip_args will be skipped when
                broadcasting f's args.

            skip_kwargs (list of keys): the names in the kwargs dict.
                The keys specified in skip_kwargs will be skipped when
                broadcasting f's kwargs.
        '''
        def mpi_map(f):
            def with_mpi(dev, *args, **kwargs):
                if pool.size <= 1 or pool.worker_status == 'R':
                    # A direct call if worker is not in pending mode
                    return f(dev, *args, **kwargs)
                else:
                    return pool.apply(_distribute_call, (None, f, dev, args, kwargs),
                                      (f.__module__, f.__name__, _dev_for_worker(dev),
                                       _update_args(args, skip_args),
                                       _update_kwargs(kwargs, skip_kwargs)))
            with_mpi.__doc__ = f.__doc__
            return with_mpi

        if fn is None:
            return mpi_map
        else:
            return mpi_map(fn)

    def _update_args(args, skip_args):
        if skip_args:
            nargs = len(args)
            args = list(args)
            for k in skip_args:
                if k == 0:
                    raise RuntimeError('First argument cannot be skipped.')
                elif k-1 < nargs:
                    # k-1 because first argument dev was excluded from args
                    args[k-1] = Message.SkippedArg
        return args

    def _update_kwargs(kwargs, skip_kwargs):
        if skip_kwargs:
            for k in skip_kwargs:
                if k in kwargs:
                    kwargs[k] = Message.SkippedArg
        return kwargs

else:
    def parallel_call(fn=None, skip_args=None, skip_kwargs=None):
        if fn is None:
            return lambda f: f
        else:
            return fn


if rank == 0:
    def _merge_yield(fn):
        def main_yield(dev, *args, **kwargs):
            for x in fn(dev, *args, **kwargs):
                yield x
            for src in range(1, pool.size):
                while True:
                    dat = comm.recv(None, source=src)
                    if dat is Message.EndOfYeild:
                        break
                    yield dat
        return main_yield
    def reduced_yield(fn=None, skip_args=None, skip_kwargs=None):
        def mpi_map(f):
            def with_mpi(dev, *args, **kwargs):
                if pool.size <= 1 or pool.worker_status == 'R':
                    return f(dev, *args, **kwargs)
                else:
                    return pool.apply(_distribute_call,
                                      (None, _merge_yield(f), dev, args, kwargs),
                                      (f.__module__, f.__name__, _dev_for_worker(dev),
                                       _update_args(args, skip_args),
                                       _update_kwargs(kwargs, skip_kwargs)))
            with_mpi.__doc__ = f.__doc__
            return with_mpi

        if fn is None:
            return mpi_map
        else:
            return mpi_map(fn)

else:
    def reduced_yield(fn=None, skip_args=None, skip_kwargs=None):
        def mpi_map(f):
            def client_yield(*args, **kwargs):
                if pool.size <= 1 or pool.worker_status == 'R':
                    return f(*args, **kwargs)
                else:
                    for x in f(*args, **kwargs):
                        comm.send(x, 0)
                    comm.send(Message.EndOfYeild, 0)
            client_yield.__doc__ = f.__doc__
            return client_yield

        if fn is None:
            return mpi_map
        else:
            return mpi_map(fn)

def _reduce_call(module, name, reg_procs, args, kwargs):
    from mpi4pyscf.tools import mpi
    result = mpi._distribute_call(module, name, reg_procs, args, kwargs)
    return mpi.reduce(result)
if rank == 0:
    def call_then_reduce(fn=None, skip_args=None, skip_kwargs=None):
        def mpi_map(f):
            def with_mpi(dev, *args, **kwargs):
                if pool.size <= 1 or pool.worker_status == 'R':
                    return f(dev, *args, **kwargs)
                else:
                    return pool.apply(_reduce_call, (None, f, dev, args, kwargs),
                                      (f.__module__, f.__name__, _dev_for_worker(dev),
                                       _update_args(args, skip_args),
                                       _update_kwargs(kwargs, skip_kwargs)))
            with_mpi.__doc__ = f.__doc__
            return with_mpi

        if fn is None:
            return mpi_map
        else:
            return mpi_map(fn)

else:
    def call_then_reduce(fn=None, skip_args=None, skip_kwargs=None):
        if fn is None:
            return lambda f: f
        else:
            return fn

def _dev_for_worker(dev):
    '''The first argument (dev) to be sent to workers'''
    if hasattr(dev, '_reg_procs'):
        return dev._reg_procs
    elif isinstance(dev, mole.Mole): # for earlier pyscf
        return dev.dumps()
    elif isinstance(dev, mole.MoleBase):  # for the current pyscf
        return dev.dumps()
    else:
        return dev


def _segment_counts(counts, p0, p1):
    counts_seg = counts - p0
    counts_seg[counts<=p0] = 0
    counts_seg[counts> p1] = p1 - p0
    return counts_seg

def prange(start, stop, step):
    '''Similar to lib.prange. This function ensures that all processes have the
    same number of steps.  It is required by alltoall communication.
    '''
    nsteps = (stop - start + step - 1) // step
    nsteps = max(comm.allgather(nsteps))
    for i in range(nsteps):
        i0 = min(stop, start + i * step)
        i1 = min(stop, i0 + step)
        yield i0, i1

def platform_info():
    def info():
        import platform
        from mpi4pyscf.tools import mpi
        info = mpi.rank, platform.node(), platform.os.getpid()
        return mpi.pool.comm.gather(info)

    if pool.size <= 1 or pool.worker_status == 'R':
        return info()
    else:
        return pool.apply(info, (), ())

class Message(enum.Enum):
    OutOfTasks = 1
    SkippedArg = 2
    NparrayToBcast = 3
    EndOfYeild = 4
