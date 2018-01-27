# Must import every submodule before suspending the slave processes
import pyscf
from distutils.version import LooseVersion
assert(LooseVersion(pyscf.__version__) >= LooseVersion('1.5'))
del(LooseVersion)

from . import lib
from . import pbc


# NOTE: suspend all slave processes at last
from .tools import mpi
if not mpi.pool.is_master():
    import sys
    import imp
# Handle global import lock for multithreading, see
#   http://stackoverflow.com/questions/12389526/import-inside-of-a-python-thread
#   https://docs.python.org/3.4/library/imp.html#imp.lock_held
# Global import lock affects the ctypes module.  It leads to deadlock when
# ctypes function is called in new threads created by threading module.
    if sys.version_info < (3,4):
        if imp.lock_held():
            imp.release_lock()

    mpi.pool.wait()

    if sys.version_info < (3,4):
        if not imp.lock_held():
            imp.acquire_lock()
    exit(0)
