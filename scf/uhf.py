#!/usr/bin/env python

from pyscf.scf import uhf

from mpi4pyscf.tools import mpi
from mpi4pyscf.scf import hf as mpi_hf

rank = mpi.rank

@mpi.register_class
class UHF(uhf.UHF, mpi_hf.SCF):

    get_jk = mpi_hf.SCF.get_jk
    get_j = mpi_hf.SCF.get_j
    get_k = mpi_hf.SCF.get_k

    def dump_flags(self):
        if rank == 0:
            uhf.UHF.dump_flags(self)
        return self
    def sanity_check(self):
        if rank == 0:
            uhf.UHF.sanity_check(self)
        return self

