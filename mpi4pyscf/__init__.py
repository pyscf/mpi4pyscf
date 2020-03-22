'''
An MPI plugin for PySCF
'''

__version__ = '0.3'

import pyscf
from distutils.version import LooseVersion
assert(LooseVersion(pyscf.__version__) >= LooseVersion('1.7'))
del(LooseVersion)

# import all pyscf submodules before suspending the slave processes
from pyscf import __all__

# NOTE: suspend all slave processes at last
from .tools import mpi
if not mpi.pool.is_master():
    import sys
    import imp
    import traceback
# Handle global import lock for multithreading, see
#   http://stackoverflow.com/questions/12389526/import-inside-of-a-python-thread
#   https://docs.python.org/3.4/library/imp.html#imp.lock_held
# Global import lock affects the ctypes module.  It leads to deadlock when
# ctypes function is called in new threads created by threading module.
    if sys.version_info < (3,4):
        if imp.lock_held():
            imp.release_lock()

    try:
        mpi.pool.wait()
    except BaseException as err:
        traceback.print_exc(err)
        mpi.comm.Abort()
        exit(1)

    if sys.version_info < (3,4):
        if not imp.lock_held():
            imp.acquire_lock()
    exit(0)
