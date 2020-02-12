from mpi4pyscf.tools import mpi

def scatter(n, m):
    import numpy
    from mpi4pyscf.tools import mpi
    mpi.INT_MAX = 7
    if mpi.rank == 0:
        arrs = [numpy.ones((n+i,m-i)) for i in range(mpi.pool.size)]
    else:
        arrs = None
    res = mpi.scatter(arrs)
    print(res.shape)

def alltoall(n, m):
    import numpy
    from mpi4pyscf.tools import mpi
    mpi.INT_MAX = 7
    arrs = [numpy.ones((n+i-mpi.rank,m-i+mpi.rank)) for i in range(mpi.pool.size)]
    res = mpi.alltoall(arrs)
    print(res.shape)

    res = mpi.alltoall(arrs, split_recvbuf=True)
    print([x.shape for x in res])

    if mpi.rank < 3:
        d1 = 3
    else:
        d1 = 1
    arrs = [numpy.zeros(s) for s in [(d1)]*mpi.pool.size]
    res = mpi.alltoall(arrs, split_recvbuf=True)
    print([x.shape for x in res])

mpi.pool.apply(alltoall, (None,), (None,))

def allgather(n, m):
    import numpy
    from mpi4pyscf.tools import mpi
    mpi.INT_MAX = 7
    arrs = numpy.ones((n-mpi.rank,m))
    res = mpi.allgather(arrs)
    print(res.shape)

def gather(n, m):
    import numpy
    from mpi4pyscf.tools import mpi
    mpi.INT_MAX = 7
    arrs = numpy.ones((n-mpi.rank,m))
    res = mpi.gather(arrs)
    if mpi.rank == 0:
        print(res.shape)

mpi.pool.apply(scatter, (20, 20), (20, 20))
mpi.pool.apply(alltoall, (20, 20), (20, 20))
mpi.pool.apply(allgather, (20, 20), (20, 20))
mpi.pool.apply(gather, (20, 20), (20, 20))
