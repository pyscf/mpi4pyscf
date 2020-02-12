#!/usr/bin/env python

from pyscf import lib
from pyscf.scf import uhf

from mpi4pyscf.tools import mpi
from mpi4pyscf.scf import hf as mpi_hf

rank = mpi.rank

@mpi.register_class
class UHF(uhf.UHF, mpi_hf.SCF):

    get_jk = mpi_hf.SCF.get_jk
    get_j = mpi_hf.SCF.get_j
    get_k = mpi_hf.SCF.get_k

    def dump_flags(self, verbose=None):
        mpi_info = mpi.platform_info()
        if rank == 0:
            uhf.UHF.dump_flags(self, verbose)
            lib.logger.debug(self, 'MPI info (rank, host, pid)  %s', mpi_info)
        return self

