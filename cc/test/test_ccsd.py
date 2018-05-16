#!/usr/bin/env python
# mpirun -np 2 python test_ccsd.py

import numpy
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import cc
from mpi4pyscf import cc as mpicc
from mpi4pyscf.tools import mpi

mol = gto.Mole()
mol.atom = [
    [2 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]
mol.basis = '6-31g'
mol.build()
mf = scf.RHF(mol)
nao = mol.nao_nr()
numpy.random.seed(1)
mf.mo_coeff = numpy.random.random((nao,nao)) - 0.5
mf.mo_occ = numpy.zeros(nao)
nocc = mol.nelectron // 2
nvir = nao - nocc
mf.mo_occ[:mol.nelectron//2] = 2

mycc = cc.CCSD(mf)
mycc.direct = True
eris = mycc.ao2mo(mf.mo_coeff)
mycc1 = mpicc.ccsd.CCSD(mf)
mycc1.ao2mo(mf.mo_coeff)
eris1 = mycc1._eris
nv = eris1.oovv.shape[2]
print(abs(numpy.asarray(eris1.oooo) - numpy.asarray(eris.oooo)).max())
print(abs(numpy.asarray(eris1.oovv) - numpy.asarray(eris.oovv[:,:,:nv])).max())
print(abs(numpy.asarray(eris1.ovvo) - numpy.asarray(eris.ovvo[:,:nv,:])).max())
print(abs(numpy.asarray(eris1.ovov) - numpy.asarray(eris.ovov[:,:nv,:])).max())

emp2, r1, r2 = mycc.init_amps(eris)
print(lib.finger(r1) - 0.20852878109950079)
print(lib.finger(r2) - 0.21333574169417541)
print(emp2 - -0.12037888088751542)

emp2, v1, v2 = mycc1.init_amps()
print(abs(v1 - r1).max())
print(abs(v2 - r2[:,:,:nv]).max())
print(emp2 - -0.12037888088751542)

t1 = numpy.random.random((nocc,nvir))
t2 = numpy.random.random((nocc,nocc,nvir,nvir))
t2 = t2 + t2.transpose(1,0,3,2)
v1, v2 = mycc.update_amps(t1, t2, eris)
print(lib.finger(v1) - 9.6029949445427079)
print(lib.finger(v2) - 4.5308876217231813)

def on_node(args):
    reg_procs, t1 = args
    from mpi4pyscf.tools import mpi
    mycc1 = mpi._registry[reg_procs[mpi.rank]]
    t2 = mycc1.t2
    eris = mycc1._eris
    t1, t2 = mycc1.update_amps(t1, t2, eris)
    t2 = mpi.gather(t2.transpose(2,3,0,1)).transpose(2,3,0,1)
    return t1, t2
mpicc.ccsd.distribute_t2_(mycc1, t2)
x1, x2 = mpi.pool.apply(on_node, (mycc1._reg_procs, t1), (mycc1._reg_procs, t1))
print(lib.finger(x1) - 9.6029949445427079)
print(lib.finger(x2) - 4.5308876217231813)


mol = gto.Mole()
mol.atom = [
    [8 , (0. , 0.     , 0.)],
    [1 , (0. , -0.757 , 0.587)],
    [1 , (0. , 0.757  , 0.587)]]
mol.basis = '6-31g'
mol.build()
mf = scf.RHF(mol).run()
mycc = cc.CCSD(mf)
mycc.kernel()
mycc1 = mpicc.ccsd.CCSD(mf)
mycc1.kernel()
print(mycc.e_tot - mycc1.e_tot)
