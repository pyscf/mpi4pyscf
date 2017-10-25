#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Parallelize k-points
'''

import pyscf.pbc.tools.pyscf_ase as pyscf_ase
import pyscf.pbc.gto as pbcgto
import pyscf.pbc.scf as pscf
from ase.lattice.cubic import Diamond
from ase.lattice import bulk

#
# mpi4pyscf provides the same code layout as pyscf
#
#from pyscf.pbc import df
from mpi4pyscf.pbc import df

ase_atom = Diamond(symbol='C', latticeconstant=3.5668)
#ase_atom = bulk('C', 'diamond', a=3.5668)

cell = pbcgto.Cell()
cell.atom = pyscf_ase.ase_atoms_to_pyscf(ase_atom)
cell.h = ase_atom.cell
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.mesh = [15]*3
cell.verbose = 5
cell.build()

kpts = cell.make_kpts([4]*3)
mf = pscf.KRHF(cell, kpts)

#
# Replace the serial DF module with MPI implementation to parallelize the J/K
# evaluation.
#
mf.with_df = df.DF(cell, kpts)
print(mf.scf())

#
# Similar replacement can be placed on MDF module
#
mf.with_df = df.MDF(cell, kpts)
print mf.scf()

