# Must import every submodule before suspending the slave processes
from espy import lib
from espy import pbc


# NOTE: suspend all slave processes at last
from espy.tools import mpi
if not mpi.pool.is_master():
    mpi.pool.wait()
    exit(0)
