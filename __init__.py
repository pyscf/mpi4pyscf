# Must import every submodule before suspending the slave processes
from . import lib
from . import pbc


# NOTE: suspend all slave processes at last
from .tools import mpi
if not mpi.pool.is_master():
    mpi.pool.wait()
    exit(0)
