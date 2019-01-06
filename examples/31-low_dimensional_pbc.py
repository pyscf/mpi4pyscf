#!/usr/bin/env python

import numpy
from pyscf import scf
#from pyscf.pbc import df as pdf
from mpi4pyscf.pbc import df as pdf
from pyscf.pbc import scf as pbchf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import tools

e = []
L = 4
cell = pbcgto.Cell()
cell.build(unit = 'B',
           a = [[L,0,0],[0,L,0],[0,0,8.]],
           mesh = [10,10,20],
           atom = 'H 0 0 0; H 0 0 1.8',
           dimension=2,
           verbose = 4,
           basis='sto3g')

mf = pbchf.KRHF(cell)
mf.with_df = pdf.AFTDF(cell)
mf.kpts = cell.make_kpts([4,4,1])
e.append(mf.kernel())

mf = pbchf.KRHF(cell, cell.make_kpts([4,4,1]))
mf = mf.density_fit(auxbasis='weigend', with_df=pdf.GDF(cell))
e.append(mf.kernel())

mf = pbchf.KRHF(cell, cell.make_kpts([4,4,1]))
mf = mf.mix_density_fit(auxbasis='weigend', with_df=pdf.MDF(cell))
e.append(mf.kernel())

mol = tools.super_cell(cell, [4,4,1]).to_mol()
mf = scf.RHF(mol)
e.append(mf.kernel()/16)

print(e)
